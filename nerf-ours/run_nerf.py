# to run this scipt: use "CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/lego.txt --ft_path logs/paper_lego/011.tar (--render_only --render_test)"
# --render_test是指定用测试视图，不加该参数，改为环绕一圈渲染，后者无法计算测试指标，因为没有GT图片

# import pydevd_pycharm
# pydevd_pycharm.settrace('166.111.81.124', port=35353, stdoutToServer=True, stderrToServer=True)

import os, sys
import pickle

import numpy as np
import imageio
import json
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from run_nerf_helpers import *
from argument_parser import config_parser
from model import NeRF, ResUNet
from render import render_rays, render_path, render

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

from tree import QuadTreeManager, get_children
from image_process import ImageProcessor
from tree_utils import RaySampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    gpu_num = torch.cuda.device_count()
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model = nn.DataParallel(model, device_ids=list(range(gpu_num)))
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        model_fine = nn.DataParallel(model_fine, device_ids=list(range(gpu_num)))
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start_epoch = 0
    start_iter = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start_epoch = ckpt['global_epoch']
        start_iter = ckpt['global_iter']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start_epoch, start_iter, grad_vars, optimizer


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    shutil.copy('run_nerf.py', os.path.join(basedir, expname, 'run_nerf_frozen.py'))

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start_epoch, start_iter, grad_vars, optimizer = create_nerf(args)
    global_epoch = start_epoch
    global_iter = start_iter

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('RENDER ONLY')
        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).cuda()
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:03d}'.format('test' if args.render_test else 'path', start_epoch))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses[i_train])
    images = torch.Tensor(images[i_train])
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # # test: a short training, use 1 image
    # images = images[:2, ...]
    # poses = poses[:2, ...]

    # tree manager
    treeManager = QuadTreeManager(H, W, K, images, poses, mseThres=0.0, max_depth=args.init_level)
    # 像加载模型一样，我们加载之前分割好的树
    tree_pkl_filename = os.path.join(args.basedir, args.expname, 'treeDivide_{:04d}.pkl'.format(global_epoch))
    if os.path.exists(tree_pkl_filename):
        with open(tree_pkl_filename, 'rb') as f:
            treeManager.quadTrees = pickle.load(f)
            treeManager.childrens = [get_children(treeManager.quadTrees[i].root) for i in range(treeManager.n_images)]
            treeManager.cur_level = global_epoch
            print("load '" + tree_pkl_filename + "'")

    # 计算max_level
    max_level = args.init_level
    i = 1
    while i <= args.n_epoch:
        # training...
        # subdivide
        if i % args.subdivide_every == 0 and i < args.n_epoch-1:
            max_level += 1
        i += 1

    # raySampler = RaySampler(treeManager.images, treeManager.dirs, treeManager.origins, max_level=max_level)
    # time1 = time.time()
    # sample_ret = raySampler.pre_gen_rays_v3(down_scale=args.rays_downscale, rand_samp_prec=args.randSamp_perc,
    #                                         dset_name=args.dset_name)
    # time2 = time.time()
    # print('pre sampling rays costs {:.2f}s.'.format(time2 - time1))


    # 开始训练
    # 如果是从头开始训练：遵循nerf的方法，先从图片中间采一部分点来训一会儿
    if global_epoch == 0:
        print('Center Cropping for 500 iters...')
        time1 = time.time()
        rays_o = []
        rays_d = []
        target_rgb = []
        H = treeManager.h
        W = treeManager.w
        dH = H // 4
        dW = W // 4
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
            ), -1).reshape(-1, 2)
        randNum = int(N_rand * 500 / treeManager.n_images)
        select_inds = np.random.choice(coords.shape[0], size=[randNum], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        for i in range(treeManager.n_images):
            rays_o.append(treeManager.origins[i][select_coords[:, 0], select_coords[:, 1]])
            rays_d.append(treeManager.dirs[i][select_coords[:, 0], select_coords[:, 1]])
            target_rgb.append(treeManager.images[i][select_coords[:, 0], select_coords[:, 1]])
        rays_o = torch.cat(rays_o, 0)
        rays_d = torch.cat(rays_d, 0)
        target_rgb = torch.cat(target_rgb, 0)

        epoch_size = rays_o.shape[0]
        batch_size = N_rand
        batch_begin = 0
        batch_end = 0
        iter = 0
        while batch_end < epoch_size:
            batch_end = min(batch_begin + batch_size, epoch_size)
            batch_origins = rays_o[batch_begin: batch_end].to(device)
            batch_dirs = rays_d[batch_begin: batch_end].to(device)
            target_s = target_rgb[batch_begin: batch_end].to(device)
            batch_rays = torch.stack([batch_origins, batch_dirs], 0)
            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, retraw=True, **render_kwargs_train)
            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            loss = img_loss
            psnr = mse2psnr(img_loss.cpu())
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0.cpu())
            loss.backward()
            optimizer.step()
            if iter % 50 == 0:
                print('{}//iter {}: coarse/loss {:.4f}, coarse/psnr {:.4f}, fine/loss {:.4f}, fine/psnr {:.4f}'
                  .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                          iter, img_loss, psnr[0], img_loss0, psnr0[0]))
            iter += 1
            batch_begin = batch_end

        print('pre Center Cropping finished. cost time: {}s.'.format(time.time() - time1))



    for epoch_id in range(global_epoch+1, args.n_epoch+1):
        print('**********************************************')
        print('**********************************************')
        print('Epoch '+str(epoch_id))
        print('**********************************************')
        print('**********************************************')

        epoch_start_time = time.time()
        print('generating rays... cur level='+str(treeManager.cur_level))
        # # 最后一轮：依旧全像素训
        if epoch_id == args.n_epoch:
            print('last epoch: use all rays to train.')
            treeManager.epoch_size = treeManager.n_images * treeManager.h * treeManager.w
            rays_o, rays_d, target_rgb = treeManager.gen_rays_v3_multiThread(down_scale=1, prob=False, last_epoch=True)
        # elif epoch_id == 1:
        #     print('first epoch: use all rays to train.')
        #     # treeManager.epoch_size = treeManager.n_images * treeManager.h * treeManager.w
        #     rays_o, rays_d, target_rgb = treeManager.gen_rays_v3_multiThread(down_scale=1, prob=False, last_epoch=True)
        # elif epoch_id < 10:
        #     print('first 10 epoch: use all rays to train.')
        #     rays_o, rays_d, target_rgb = treeManager.gen_rays_v3_multiThread(down_scale=1, prob=False, last_epoch=True)
        else:
            # rays_o, rays_d, target_rgb = treeManager.gen_rays_v4(sample_ret, down_scale=1, debug=False)

            # print('after 10 epoch: use prob rays to train.')
            rays_o, rays_d, target_rgb = treeManager.gen_rays_v3_multiThread(down_scale=1, prob=False, randSamp_proc=args.randSamp_perc, last_epoch=False)

            # rays_o, rays_d, target_rgb = treeManager.gen_rays_v3_multiThread(down_scale=1, prob=False, last_epoch=False)

        time2 = time.time()
        print('shuffling rays costs {:.2f}s.'.format(time2 - epoch_start_time))
        print('training rays num: '+str(rays_o.shape[0]))


        epoch_size = rays_o.shape[0]
        batch_size = N_rand
        batch_begin = 0
        batch_end = 0
        iter = 0

        # 收集一个epoch所有的gt和pred，用于该轮结束后调整四叉树用
        rgb_gt_collect = []
        rgb_pred_collect = []
        while batch_end < epoch_size:
            batch_end = min(batch_begin + batch_size, epoch_size)
            batch_origins = rays_o[batch_begin: batch_end].to(device)
            batch_dirs = rays_d[batch_begin: batch_end].to(device)
            target_s = target_rgb[batch_begin: batch_end].to(device)

            batch_rays = torch.stack([batch_origins, batch_dirs], 0)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, retraw=True,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss.cpu())

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0.cpu())

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_iter / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            rgb_gt_collect.append(target_s.cpu().detach())
            rgb_pred_collect.append(rgb.cpu().detach())

            global_iter += 1
            if iter % 400 == 0:
                print('{}//iter {}: coarse/loss {:.4f}, coarse/psnr {:.4f}, fine/loss {:.4f}, fine/psnr {:.4f}'
                      .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                              iter, img_loss, psnr[0], img_loss0, psnr0[0]))

                # go to next iter
            iter += 1
            batch_begin = batch_end

        print('{}//total: {} iters.'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iter))
        # 隔若干轮细化一次四叉树
        if args.subdivide_every > 0 and epoch_id % args.subdivide_every == 0 and epoch_id < args.n_epoch - 1:
            time1 = time.time()
            rgb_gt = torch.cat(rgb_gt_collect, 0)
            rgb_pred = torch.cat(rgb_pred_collect, 0)
            # treeManager.adjust_tree(rgb_gt, rgb_pred, thres=0.005, debug=False)
            treeManager.adjust_tree_multiThread(rgb_gt, rgb_pred, thres=args.subdivide_thres, debug=False)
            time2 = time.time()
            print('adjust quadTree cost {:.2f}s.'.format(time2 - time1))



        # 隔一个epoch保存一次模型
        path = os.path.join(basedir, expname, '{:03d}.tar'.format(epoch_id))
        torch.save({
            'global_epoch': epoch_id,
            'global_iter': global_iter,
            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
            'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)

        tree_pkl_filename = os.path.join(args.basedir, args.expname, 'treeDivide_{:04d}.pkl'.format(epoch_id))
        with open(tree_pkl_filename, 'wb') as f:
            pickle.dump(treeManager.quadTrees, f)

        print('one step finished. cost time: {}s.'.format(int(time.time()-epoch_start_time)))

        



if __name__=='__main__':
    # torch.set_default_tensor_type('torch.FloatTensor')

    start_time = time.time()

    train()

    end_time = time.time()
    print('train complete. time={:.1f}s.'.format(end_time - start_time))