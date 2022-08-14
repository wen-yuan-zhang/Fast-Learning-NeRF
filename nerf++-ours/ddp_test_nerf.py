
"""
CUDA_VISIBLE_DEVICES=2 python ddp_test_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt --render_splits test
"""

# import pydevd_pycharm
# pydevd_pycharm.settrace('166.111.81.124', port=13214, stdoutToServer=True, stderrToServer=True)

import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b, compute_ssim
import imageio
from ddp_train_nerf import config_parser, setup_logger, create_nerf, intersect_sphere, sample_pdf
import logging
from collections import OrderedDict


logger = logging.getLogger(__package__)

import lpips
lpips_vgg = lpips.LPIPS(net="vgg").eval().cuda()


def ddp_test_nerf(args):
    ###### set up logger
    logger = logging.getLogger(__package__)

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(0).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 8960

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(0, args)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth)
        mean_psnr = []
        mean_ssim = []
        mean_lpips = []
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            ret = render_single_image(models, ray_samplers[idx], args.chunk_size)
            dt = time.time() - time0

            logger.info('Rendered {} in {} seconds'.format(fname, dt))

            # only save last level
            im = ret[-1]['rgb'].numpy()
            # compute psnr if ground-truth is available
            if ray_samplers[idx].img_path is not None:
                gt_im = ray_samplers[idx].get_img()
                psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                gt_im = torch.tensor(gt_im)
                im = torch.tensor(im)
                ssim = compute_ssim(gt_im, im).item()
                lpips_i = lpips_vgg(gt_im.cuda().permute([2, 0, 1]).contiguous(),
                                    im.cuda().permute([2, 0, 1]).contiguous(), normalize=True).item()
                logger.info('{}: psnr={}, ssim={}, lpips={}'.format(fname, psnr, ssim, lpips_i))
                mean_psnr.append(psnr)
                mean_ssim.append(ssim)
                mean_lpips.append(lpips_i)

            im = to8b(im.numpy())
            imageio.imwrite(os.path.join(out_dir, fname), im)

            # im = ret[-1]['fg_rgb'].numpy()
            # im = to8b(im)
            # imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)
            #
            # im = ret[-1]['bg_rgb'].numpy()
            # im = to8b(im)
            # imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)
            #
            # im = ret[-1]['fg_depth'].numpy()
            # im = colorize_np(im, cmap_name='jet', append_cbar=True)
            # im = to8b(im)
            # imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)
            #
            # im = ret[-1]['bg_depth'].numpy()
            # im = colorize_np(im, cmap_name='jet', append_cbar=True)
            # im = to8b(im)
            # imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

            torch.cuda.empty_cache()

        results = 'mean PSNR: {}\nmean SSIM: {}\nmean LPIPS: {}'\
            .format(np.mean(mean_psnr), np.mean(mean_ssim), np.mean(mean_lpips))
        print(results)
        with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
            f.write(results)



def render_single_image(models, ray_sampler, chunk_size):
    ##### parallel rendering of a single image
    ray_batch = ray_sampler.get_all()

    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch[key] = ray_batch[key].cuda()

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        min_depth = ray_batch_split['min_depth'][s]

        dots_sh = list(ray_d.shape[:-1])
        for m in range(models['cascade_level']):
            net = models['net_{}'.format(m)]
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                    [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).cuda()

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples
                del bg_weights
                del bg_depth_mid
                del bg_depth_samples
                torch.cuda.empty_cache()

            with torch.no_grad():
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth)

            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()

    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0).reshape(ray_sampler.H, ray_sampler.W, -1)

    # # merge results from different processes
    # ret_merge_rank = [OrderedDict() for _ in range(len(ret_merge_chunk))]
    # for m in range(len(ret_merge_chunk)):
    #     for key in ret_merge_chunk[m]:
    #         # generate tensors to store results from other processes
    #         sh = list(ret_merge_chunk[m][key].shape[1:])
    #         ret_merge_rank[m][key] = [torch.zeros(*[size, ] + sh, dtype=torch.float32) for size in rank_split_sizes]
    #         torch.distributed.gather(ret_merge_chunk[m][key], ret_merge_rank[m][key])
    #         ret_merge_rank[m][key] = torch.cat(ret_merge_rank[m][key], dim=0).reshape(
    #             (ray_sampler.H, ray_sampler.W, -1)).squeeze()
    #         # print(m, key, ret_merge_rank[m][key].shape)

    # only rank 0 program returns
    return ret_merge_chunk


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))

    ddp_test_nerf(args)



if __name__ == '__main__':
    setup_logger()
    test()

