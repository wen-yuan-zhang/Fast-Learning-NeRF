"""
linux comman line:
CUDA_VISIBLE_DEVICES=1 python ddp_train_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt
"""

# import pydevd_pycharm
# pydevd_pycharm.settrace('166.111.81.124', port=13214, stdoutToServer=True, stderrToServer=True)

import torch
import torch.nn as nn
import torch.optim
import torch.distributed
import torch.multiprocessing
import os
import time
import numpy as np
import logging
import json
import shutil
import pickle
from collections import OrderedDict

from ddp_model import NerfNetWithAutoExpo
from data_loader_split import load_data_split
from tree_utils import RaySampler
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, TINY_NUMBER
from tree import QuadTreeManager, get_children
from image_process import ImageProcessor

logger = logging.getLogger(__package__)
device = torch.device('cuda:0')


def setup_logger():
    # create logger
    logger = logging.getLogger(__package__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception('Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER      # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00       # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
    denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples


def create_nerf(rank, args):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777)
    gpu_num = torch.cuda.device_count()

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]
    for m in range(models['cascade_level']):
        img_names = None
        if args.optim_autoexpo:
            # load training image names for autoexposure
            f = os.path.join(args.basedir, args.expname, 'train_images.json')
            with open(f) as file:
                img_names = json.load(file)
        net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).cuda()
        net = nn.DataParallel(net, device_ids=list(range(gpu_num)))
        # net = DDP(net, device_ids=[rank], output_device=rank)
        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = net
        models['optim_{}'.format(m)] = optim

    start = 0

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]
    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])
    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        start = path2iter(fpath)
        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)
        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
                models[name].load_state_dict(to_load[name])

    return start, models


def ddp_train_nerf(args):

    ###### set up logger
    logger = logging.getLogger(__package__)
    # setup_logger()

    ###### decide chunk size according to gpu memory
    logger.info('gpu_mem: {}'.format(torch.cuda.get_device_properties(0).total_memory))
    if torch.cuda.get_device_properties(0).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        # batch size=1920占2张卡，对应500000个iter在226张图片上一共训练7.94≈8轮，每100iter用55秒，共64h
        # batch size=2880占3张卡，对应500000个iter在226张图片上一共训练≈12轮，每100iter用45秒，一个epoch用时5.2h，感觉训5~6轮就够了
        # 从第1轮后开始细分，分到第4轮结束，第5轮开始前全像素训，初始时4层，结束时6层(level=7)。
        args.N_rand = args.batch_size
        args.chunk_size = 4096

    ###### Create log dir and copy the config file
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(args.basedir, args.expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    shutil.copyfile('ddp_train_nerf.py', os.path.join(args.basedir, args.expname, 'ddp_train_nerf_frozen.txt'))

    ray_samplers = load_data_split(args.datadir, args.scene, split='train',
                                   try_load_min_depth=args.load_min_depth)
    # val_ray_samplers = load_data_split(args.datadir, args.scene, split='validation',
    #                                    try_load_min_depth=args.load_min_depth, skip=args.testskip)

    # write training image names for autoexposure
    if args.optim_autoexpo:
        f = os.path.join(args.basedir, args.expname, 'train_images.json')
        with open(f, 'w') as file:
            img_names = [ray_samplers[i].img_path for i in range(len(ray_samplers))]
            json.dump(img_names, file, indent=2)

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(0, args)

    treeManager = QuadTreeManager(ray_samplers, mseThres=0.0, max_depth=args.init_level)      # 分4块时，depth=2, level=1

    # # 像加载模型一样，我们加载之前分割好的树
    # tree_pkl_filename = os.path.join(args.basedir, args.expname, 'treeDivide_{:04d}.pkl'.format(start))
    # if os.path.exists(tree_pkl_filename):
    #     with open(tree_pkl_filename, 'rb') as f:
    #         treeManager.quadTrees = pickle.load(f)
    #         treeManager.childrens = [get_children(treeManager.quadTrees[i].root) for i in range(treeManager.n_images)]
    #         treeManager.cur_level = start + args.init_level + 2
    #         print("load '" + tree_pkl_filename + "'")

    # 计算max_level
    # max_level = args.init_level
    # i = 1
    # while i <= args.n_epoch:
    #     # training...
    #     # subdivide
    #     if i % args.subdivide_every == 0 and i < args.n_epoch-1:
    #         max_level += 1
    #     i += 1

    # raySampler = RaySampler(ray_samplers, max_level=max_level)
    # time1 = time.time()
    # sample_ret = raySampler.pre_gen_rays_v3(down_scale=args.rays_downscale, rand_samp_prec=args.randSamp_perc,
    #                                         dset_name=args.dset_name)
    # time2 = time.time()
    # print('pre sampling rays costs {:.2f}s.'.format(time2 - time1))

    ##### important!!!
    # np.random.seed(777)
    # torch.manual_seed(777)


    # start training
    for epoch_id in range(start+1, args.n_epoch+1):
        print('**********************************************')
        print('**********************************************')
        print('Epoch '+str(epoch_id))
        print('**********************************************')
        print('**********************************************')

        time0 = time.time()
        # 最后一轮依旧全像素训
        print('generating rays...')
        if epoch_id == args.n_epoch:
            print('last epoch: use all rays to train.')
            treeManager.epoch_size = treeManager.n_images * treeManager.h * treeManager.w
            rays_o, rays_d, target_rgb = treeManager.gen_rays_v3_multiThread(down_scale=args.rays_downscale, prob=False, last_epoch=True)
        else:
            rays_o, rays_d, target_rgb = treeManager.gen_rays_v3_multiThread(down_scale=args.rays_downscale, prob=True, rand=args.randSamp_perc, last_epoch=False, debug=True)
            # rays_o, rays_d, target_rgb = treeManager.gen_rays_v4(sample_ret, down_scale=1, debug=False)
        # rays_o, rays_d, target_rgb = processor.gen_rays(down_scale=1, debug=False)
        print('generating rays finished. cost time: {:.2f}s.'.format(time.time() - time0))
        print('training rays num: '+str(rays_o.shape[0]))

        # for test: 设置一个小batch的数据来训练
        # truncation = 9600
        # rays_o = rays_o[:truncation, ...]
        # rays_d = rays_d[:truncation, ...]
        # target_rgb = target_rgb[:truncation, ...]
        rgb_pred_collect = train_step(models, rays_o, rays_d, target_rgb, args)

        # 隔一轮细分一次四叉树。第一轮先不分
        if epoch_id % args.subdivide_every == 0 and epoch_id < args.n_epoch-1:
            print('training epoch finished. start subdividing quadtrees...')
            time1 = time.time()
            # treeManager.adjust_tree(rgb_gt, rgb_pred, thres=0.005, debug=False)
            treeManager.adjust_tree_multiThread(target_rgb, rgb_pred_collect, thres=args.subdivide_thres, debug=False)
            dt = time.time() - time1
            print('adjust quadTree cost {:.2f}s.'.format(dt))

        # saving checkpoints and logging
        fpath = os.path.join(args.basedir, args.expname, 'model_{:04d}.pth'.format(epoch_id))
        to_save = OrderedDict()
        for m in range(models['cascade_level']):
            name = 'net_{}'.format(m)
            to_save[name] = models[name].state_dict()

            name = 'optim_{}'.format(m)
            to_save[name] = models[name].state_dict()
        torch.save(to_save, fpath)

        # tree_pkl_filename = os.path.join(args.basedir, args.expname, 'treeDivide_{:04d}.pkl'.format(epoch_id))
        # with open(tree_pkl_filename, 'wb') as f:
        #     pickle.dump(treeManager.quadTrees, f)


        ### end of core optimization loop
        dt = time.time() - time0
        print('one step finished. cost time: {}s.'.format(int(dt)))


def train_step(models, rays_o, rays_d, target_rgb, args):

    scalars_to_log = OrderedDict()
    ### Start of core optimization loop
    epoch_size = rays_o.shape[0]
    batch_size = args.batch_size
    batch_begin = 0
    batch_end = 0
    iter = 0
    rgb_pred_collect = []
    while batch_end < epoch_size:
        batch_end = min(batch_begin + batch_size, epoch_size)
        batch_origins = rays_o[batch_begin: batch_end].cuda()
        batch_dirs = rays_d[batch_begin: batch_end].cuda()
        batch_rgb = target_rgb[batch_begin: batch_end].cuda()

        ray_batch = {'ray_o': batch_origins, 'ray_d': batch_dirs, 'rgb': batch_rgb,
                     'min_depth': 1e-4*torch.ones_like(batch_dirs[..., 0])}

        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        all_rets = []  # results on different cascade levels
        for m in range(models['cascade_level']):
            optim = models['optim_{}'.format(m)]
            net = models['net_{}'.format(m)]

            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_batch['ray_o'], ray_batch['ray_d'])  # [...,]
                fg_near_depth = ray_batch['min_depth']  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                fg_depth = perturb_samples(fg_depth)  # random perturbation during training

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                    [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).cuda()
                bg_depth = perturb_samples(bg_depth)  # random perturbation during training
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=False)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=False)  # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

            optim.zero_grad()
            ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth,
                      img_name=None)
            all_rets.append(ret)

            rgb_gt = ray_batch['rgb'].cuda()
            if 'autoexpo' in ret:
                scale, shift = ret['autoexpo']
                scalars_to_log['level_{}/autoexpo_scale'.format(m)] = scale.item()
                scalars_to_log['level_{}/autoexpo_shift'.format(m)] = shift.item()
                # rgb_gt = scale * rgb_gt + shift
                rgb_pred = (ret['rgb'] - shift) / scale
                rgb_loss = img2mse(rgb_pred, rgb_gt)
                loss = rgb_loss + args.lambda_autoexpo * (torch.abs(scale - 1.) + torch.abs(shift))
            else:
                rgb_loss = img2mse(ret['rgb'], rgb_gt)
                loss = rgb_loss
            scalars_to_log['level_{}/loss'.format(m)] = rgb_loss.item()
            scalars_to_log['level_{}/psnr'.format(m)] = mse2psnr(rgb_loss.item())
            loss.backward()
            optim.step()

        # collect rgb of every iter
        rgb_pred_collect.append(ret['rgb'].cpu().detach())
        # write log
        if iter % 400 == 0:
            print('{}//iter {}: level1/loss {:.4f}, level1/psnr {:.4f}, level2/loss {:.4f}, level2/psnr {:.4f}'
                  .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                          iter, scalars_to_log['level_0/loss'], scalars_to_log['level_0/psnr'],
                          scalars_to_log['level_1/loss'], scalars_to_log['level_1/psnr']))
        # go to next iter
        iter += 1
        batch_begin = batch_end

    print('{}//total: {} iters.'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iter))

    # clean unused memory
    torch.cuda.empty_cache()

    rgb_pred_collect = torch.cat(rgb_pred_collect, 0)
    return rgb_pred_collect



def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    # dataset options
    parser.add_argument("--datadir", type=str, default=None, help='input data directory')
    parser.add_argument("--scene", type=str, default=None, help='scene name')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # model size
    parser.add_argument("--netdepth", type=int, default=8, help='layers in coarse network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer in coarse network')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    # ours added
    parser.add_argument("--init_level", type=int, default=3, help='init quadtree subdivide level, block number is 4^(level-1).')
    parser.add_argument("--subdivide_every", type=int, default=1, help='subdivide quadtrees every x epochs')
    parser.add_argument("--subdivide_thres", type=float, default=0.015)
    parser.add_argument("--rays_downscale", type=int, default=1)
    parser.add_argument("--randSamp_perc", type=float, default=0.5)
    parser.add_argument("--dset_name", type=str, default='Truck')

    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    # batch size
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 2,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--batch_size", type=int, default=2880)

    # iterations
    parser.add_argument("--N_iters", type=int, default=250001, help='number of iterations')
    parser.add_argument("--n_epoch", type=int, default=6, help='number of epochs')

    # render only
    parser.add_argument("--render_splits", type=str, default='test', help='splits to render')

    # cascade training
    parser.add_argument("--cascade_level", type=int, default=2, help='number of cascade levels')
    parser.add_argument("--cascade_samples", type=str, default='64,64', help='samples at each level')

    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1', help='number of processes')

    # optimize autoexposure
    parser.add_argument("--optim_autoexpo", action='store_true', help='optimize autoexposure parameters')
    parser.add_argument("--lambda_autoexpo", type=float, default=1., help='regularization weight for autoexposure')

    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=5000,
                        help='decay learning rate by a factor every specified number of steps')
    # rendering options
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--load_min_depth", action='store_true', help='whether to load min depth')
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))

    ddp_train_nerf(args)


if __name__ == '__main__':
    setup_logger()
    train()


