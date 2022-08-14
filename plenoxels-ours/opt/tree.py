import torch
import math
import numpy as np
from cv2 import cv2
import threading
import torch.nn.functional as F
from tqdm import tqdm
from typing import List
import threadpool
import os
import matplotlib.pyplot as plt


from util.util import Rays
from image_process import ImageProcessor


class QuadTreeNode:
    """四叉树节点"""
    def __init__(self, x0, y0, x1, y1):
        """
        :param x0, y0, x1, y1: 节点区域左上角、右下角坐标
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.children = []

    def get_error(self, img):
        """
        计算节点块内的像素均方误差
        :param img: [h, w, 3]
        :return:
        """
        # 如果划分的块面积小于一个像素点^2，怎么办？
        x0 = math.ceil(self.x0)
        x1 = math.floor(self.x1)
        y0 = math.ceil(self.y0)
        y1 = math.floor(self.y1)
        pixels = img[x0: x1, y0:y1, :].numpy()

        red = pixels[:, :, 0]
        r_avg = np.mean(red)
        r_mse = np.square(np.subtract(red, r_avg)).mean()
        blue = pixels[:, :, 1]
        b_avg = np.mean(blue)
        b_mse = np.square(np.subtract(blue, b_avg)).mean()
        green = pixels[:, :, 2]
        g_avg = np.mean(green)
        g_mse = np.square(np.subtract(green, g_avg)).mean()

        return r_mse + b_mse + g_mse

    def subdivide_once(self):
        """
        在当前节点上进行一次划分，新建立四个孩子节点，替代现在的self.children=[]
        """
        midx = (self.x0 + self.x1) / 2
        midy = (self.y0 + self.y1) / 2
        # 左上
        node1 = QuadTreeNode(self.x0, self.y0, midx, midy)
        # 左下
        node2 = QuadTreeNode(midx, self.y0, self.x1, midy)
        # 右上
        node3 = QuadTreeNode(self.x0, midy, midx, self.y1)
        # 右下
        node4 = QuadTreeNode(midx, midy, self.x1, self.y1)

        self.children = [node1, node2, node3, node4]

    @property
    def area(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def __str__(self):
        return "({:.1f}, {:.1f}), ({:.1f}, {:.1f})".format(self.x0, self.y0, self.x1, self.y1)


class QuadTree:
    """四叉树"""
    def __init__(self, image: torch.Tensor, stdThres: float, max_depth: int):
        """
        :param image: [h, w, 3]
        :param stdThres:
        """
        self.H, self.W = image.shape[:2]
        self.threshold = stdThres
        self.image = image
        self.root = QuadTreeNode(0, 0, self.H, self.W)
        self.init_subdivide_v1(max_depth)
        self.minArea = self.H * self.W / (4 ** (max_depth-1))      # 存储当前最小叶子节点的面积，给QuadTreeManager.gen_rays_v3用

    def init_subdivide_v1(self, max_depth):
        """
        第一次划分是均匀的四叉树，横着竖着从中间切两刀
        """
        recursive_subdivide(self.root, self.threshold, self.image, 1, max_depth)

    def init_subdivide_v2(self, max_depth):
        # 第一次不均匀划分，判断划分线两侧error差异越大越好。横竖各分一刀，
        # naive implementation: 在六等分线中选一条
        # 贪心：水平方向和竖直方向独立地选取
        horizontal_split_pos = [int(self.H / 6 * i) for i in range(1, 6)]
        max_error_diff = -1
        best_hori_pos = -1
        for pos in horizontal_split_pos:
            top_node = QuadTreeNode(0, 0, pos, self.W)
            top_error = top_node.get_error(self.image)
            bot_node = QuadTreeNode(pos, 0, self.H, self.W)
            bot_error = bot_node.get_error(self.image)
            error_diff = abs(top_error - bot_error)
            if error_diff > max_error_diff:
                max_error_diff = error_diff
                best_hori_pos = pos

        vertical_split_pos = [int(self.W / 6 * i) for i in range(1, 6)]
        max_error_diff = -1
        best_verti_pos = -1
        for pos in vertical_split_pos:
            left_node = QuadTreeNode(0, 0, self.H, pos)
            left_error = left_node.get_error(self.image)
            right_node = QuadTreeNode(0, pos, self.H, self.W)
            right_error = right_node.get_error(self.image)
            error_diff = abs(left_error - right_error)
            if error_diff > max_error_diff:
                max_error_diff = error_diff
                best_verti_pos = pos

        # 左上
        node1 = QuadTreeNode(0, 0, best_hori_pos, best_verti_pos)
        recursive_subdivide(node1, self.threshold, self.image, 2, max_depth)
        # 左下
        node2 = QuadTreeNode(best_hori_pos, 0, self.H, best_verti_pos)
        recursive_subdivide(node2, self.threshold, self.image, 2, max_depth)
        # 右上
        node3 = QuadTreeNode(0, best_verti_pos, best_hori_pos, self.W)
        recursive_subdivide(node3, self.threshold, self.image, 2, max_depth)
        # 右下
        node4 = QuadTreeNode(best_hori_pos, best_verti_pos, self.H, self.W)
        recursive_subdivide(node4, self.threshold, self.image, 2, max_depth)

        self.root.children = [node1, node2, node3, node4]


    def visualize_tree(self, filename):
        imgc = self.image.numpy().copy() * 255.0
        c = get_children(self.root)
        for n in c:
            # Draw a rectangle
            imgc = cv2.rectangle(imgc, (int(n.y0), int(n.x0)), (int(n.y1), int(n.x1)), (0, 0, 0), 1)

        cv2.imwrite('debug/'+filename+'.jpg', imgc[:, :, [2,1,0]])
        return imgc


class QuadTreeManager:
    """四叉树管理器：包括输入图像和光线、划分四叉树、获得每个叶子节点的采样光线和GT color等"""
    def __init__(self, dataset, mseThres=0.1, max_depth=3, prob_scale=50):
        """
        :param dataset: one of NeRFDataset, LLFFDataset, NSVFDataset, CO3DDataset
        :param self.images: torch.Tensor([n_images, h, w, 3]
        :param max_depth: 指定树的最大层数。根节点也算一层
        """
        self.n_images, self.h, self.w = dataset.n_images, dataset.h_full, dataset.w_full
        rays = dataset.rays
        self.dataset = dataset
        self.init_epoch_size = dataset.epoch_size
        self.images = rays.gt.reshape(self.n_images, self.h, self.w, -1)
        self.dirs = rays.dirs.reshape(self.n_images, self.h, self.w, -1)
        self.origins = rays.origins.reshape(self.n_images, self.h, self.w, -1)

        self.processor = ImageProcessor(dataset, scale=prob_scale)

        # debug
        # self.n_images = 4
        self.quadTrees = []
        print('init QuadTrees...')
        for i in tqdm(range(self.n_images)):
            self.quadTrees.append(QuadTree(self.images[i], mseThres, max_depth))
        print('{} QuadTrees created.'.format(self.n_images))
        self.cur_level = max_depth  # 当前四叉树的最深深度，留着给gen_rays_v4用

        self.childrens = [get_children(self.quadTrees[i].root) for i in range(self.n_images)]

    def visualize_subdivide(self, tree_id=-1, filename_prefix='tree_subdivide'):
        """
        可视化四叉树的划分情况
        :param tree_id: 如果指定为-1，则可视化所有树
        :return:
        """
        if tree_id == -1:
            for i in range(self.n_images):
                self.quadTrees[i].visualize_tree('tree_subdivide_'+str(i))
        else:
            self.quadTrees[tree_id].visualize_tree('tree_subdivide_'+str(tree_id))

    def visualize_image_split_and_sample_points(self, img_id, selected_pixel):
        """
        同时可视化图片上的物体、四叉树划分和在树上的采样点
        """
        imgc = self.images[img_id].numpy().copy() * 255.0
        c = get_children(self.quadTrees[img_id].root)
        for n in c:
            # Draw a rectangle
            imgc = cv2.rectangle(imgc, (int(n.y0), int(n.x0)), (int(n.y1), int(n.x1)), (0, 0, 0), 1)

        for x, y in selected_pixel.numpy():
            imgc = cv2.circle(imgc, (int(y), int(x)), 1, (255, 0, 0), -1)

        cv2.imwrite('debug/object_and_split_and_sample_points_{}.jpg'.format(img_id), imgc[:, :, [2, 1, 0]])

    def visualize_split_and_sample_points(self, img_id, selected_pixel):
        # 同时可视化四叉树划分和在树上的采样点，图片不显示，防止挡住采样点
        imgc = np.ones_like(self.images[img_id].numpy().copy()) * 255.0
        c = get_children(self.quadTrees[img_id].root)
        for x, y in selected_pixel.numpy():
            imgc = cv2.circle(imgc, (int(y), int(x)), 0, (255, 0, 0), -1)
        for n in c:
            # Draw a rectangle
            imgc = cv2.rectangle(imgc, (int(n.y0), int(n.x0)), (int(n.y1), int(n.x1)), (0, 0, 0), 1)

        # TODO
        debug_dir = 'debug/train/Playground'
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, 'split_and_sample_points_{}_{}childrens.jpg'.format(str(img_id), len(self.childrens[img_id]))), imgc[:, :, [2, 1, 0]])

    def visualize_image_split_and_mean_loss(self, img_id, loss):
        """
        可视化图片和每个叶子节点上面的loss。loss用半透明颜色画在上面，颜色越深表示平均loss越大
        :param img_id: 这张图片的id
        :param img_i_idx: 打乱光线顺序后的idx
        :param loss:
        :return:
        """
        imgc = self.images[img_id].numpy().copy() * 255.0

        img_i_idx = torch.where(self.result_leaf_id[:, 0] == img_id)
        img_i_leaves = self.result_leaf_id[img_i_idx][:, 1]
        img_i_loss = loss[img_i_idx]

        linear_colors = plt.get_cmap('BuGn')
        def loss2color(cur_loss, max_loss):
            color_scale = (cur_loss / max_loss).item()
            float2color = linear_colors(color_scale)
            intcolor = [int(float2color[i] * 255) for i in range(3)]
            return tuple(intcolor)

        # 先找到叶子节点上平均loss的最大值
        img_max_mean_loss = -1
        for leaf_i, leaf in enumerate(self.childrens[img_id]):
            leaf_i_idx = torch.where(img_i_leaves == leaf_i)
            leaf_i_loss = img_i_loss[leaf_i_idx]
            if leaf_i_loss.mean() > img_max_mean_loss:
                img_max_mean_loss = leaf_i_loss.mean()

        blank = np.zeros_like(self.images[img_id].numpy())
        for leaf_i, leaf in enumerate(self.childrens[img_id]):
            leaf_i_idx = torch.where(img_i_leaves == leaf_i)
            leaf_i_loss = img_i_loss[leaf_i_idx]
            leaf_mean_loss = leaf_i_loss.mean()

            cv2.rectangle(blank, (int(leaf.y0), int(leaf.x0)), (int(leaf.y1), int(leaf.x1)), loss2color(leaf_mean_loss, img_max_mean_loss), -1)

        scale = 0.2
        imgc = cv2.addWeighted(imgc, scale, blank, 1-scale, 0)

        for n in self.childrens[img_id]:
            # Draw a rectangle
            imgc = cv2.rectangle(imgc, (int(n.y0), int(n.x0)), (int(n.y1), int(n.x1)), (0, 0, 0), 1)

        # TODO
        debug_dir = 'debug/train/Playground'
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, 'split_and_mean_loss_{}_{}childrens.jpg'.format(str(img_id), len(self.childrens[img_id]))), imgc[:, :, [2, 1, 0]])

    def gen_rays_v3(self, down_scale=16, debug=False):
        """
        在v2中，我们在每个叶子节点里都取固定数量的光线，保证图片上的总光线数等于plenoxel中的光线数。
        在这一版中，我们考虑在非最底层叶子节点处采样更少的光线：如果某个节点在某一次adjust时没有被分（节点面积<当前最小节点面积），
        那么以后每次都只在这个节点里采10条光线。因为没有被细分代表这个块已经学好了，没必要在上面再投入更多的光线。
        :param down_scale: 下采样光线的倍数
        :param debug: 如果设置为true，在处理时保存所有采样像素点坐标，用于可视化
        """

        ray_num_per_image = self.dataset.epoch_size / self.n_images / down_scale
        scale = 1000
        # leay_ray_num[i]对应第i张图片应该在每个最小叶子节点里采多少条光线
        leaf_ray_num = [int(ray_num_per_image / len(self.childrens[i])) for i in range(self.n_images)]
        ray_num_per_pixel = ray_num_per_image / self.h / self.w
        # 构造光线用
        result_rgb = []
        result_dir = []
        result_origin = []
        # 后面调整四叉树时用
        self.result_leaf_id = []                         # [N, 2], N采样像素个数，2分别表示图片id、叶子节点编号
        # debug时，visualize用
        result_selected_pixel = []

        for img_i in tqdm(range(self.n_images)):
            for leaf_id, child in enumerate(self.childrens[img_i]):
                if child.area > self.quadTrees[img_i].minArea + 0.01:
                    ray_num = 10
                else:
                    ray_num = int(child.area * ray_num_per_pixel)       # 最底层叶子节点的光线数由面积*平均每个像素采样点数来决定

                selected_pixel_x = torch.randint(int(child.x0 * scale), int((child.x1 - 0.01) * scale),
                                                 (ray_num,)) / scale
                selected_pixel_y = torch.randint(int(child.y0 * scale), int((child.y1 - 0.01) * scale),
                                                 (ray_num,)) / scale
                seleted_pixel = torch.stack([selected_pixel_x, selected_pixel_y], 1)

                if debug:
                    result_selected_pixel.append(seleted_pixel)

                # 归一化到[-1, 1]
                normalized_pixel = seleted_pixel / torch.Tensor([self.h / 2, self.w / 2]) - 1
                normalized_pixel = normalized_pixel.unsqueeze(0).unsqueeze(2)  # [1, leaf_ray_num, 1, 2]

                # [2 or 3, leaf_num]
                interpolated_rgb = F.grid_sample(self.images[img_i:img_i + 1].permute(0, 3, 1, 2),
                                                 normalized_pixel).squeeze()
                interpolated_dir = F.grid_sample(self.dirs[img_i:img_i + 1].permute(0, 3, 1, 2),
                                                 normalized_pixel).squeeze()
                interpolated_origin = F.grid_sample(self.origins[img_i:img_i + 1].permute(0, 3, 1, 2),
                                                    normalized_pixel).squeeze()

                result_rgb.append(interpolated_rgb)
                result_dir.append(interpolated_dir)
                result_origin.append(interpolated_origin)
                self.result_leaf_id.append(torch.Tensor([[img_i, leaf_id]]).repeat([ray_num, 1]))

            if debug:
                result_selected_pixel = torch.cat(result_selected_pixel, 0)     # [N, 2]
                self.visualize_split_and_sample_points(img_i, result_selected_pixel)
                result_selected_pixel = []

        result_rgb = torch.cat(result_rgb, 1).transpose(0, 1)
        ray_num = result_rgb.shape[0]
        shuffle_index = torch.randperm(ray_num)
        result_rgb = result_rgb[shuffle_index].contiguous()
        result_dir = torch.cat(result_dir, 1).transpose(0, 1)[shuffle_index].contiguous()
        result_origin = torch.cat(result_origin, 1).transpose(0, 1)[shuffle_index].contiguous()
        # [N, 2], N采样像素个数，2分别表示图片id、叶子节点编号
        self.result_leaf_id = torch.cat(self.result_leaf_id, 0)[shuffle_index]

        return Rays(result_origin, result_dir, result_rgb).to(device=self.dataset.device)

    def gen_rays_v3_1(self, down_scale=16, prob=True, debug=False):
        """
        gen_rays_v3的另一个版本，在这里我们只采样整数像素位置的光线
        :param down_scale: 下采样光线的倍数
        :param prob: 在每个叶子节点内概率采样or随机采样
        :param debug: 如果设置为true，在处理时保存所有采样像素点坐标，用于可视化
        """

        ray_num_per_image = self.dataset.epoch_size / self.n_images / down_scale
        scale = 1000
        ray_num_per_pixel = ray_num_per_image / self.h / self.w
        # 构造光线用
        result_rgb = []
        result_dir = []
        result_origin = []
        # 后面调整四叉树时用
        self.result_leaf_id = []                         # [N, 2], N采样像素个数，2分别表示图片id、叶子节点编号
        # debug时，visualize用
        result_selected_pixel = []

        for img_i in tqdm(range(self.n_images)):
            for leaf_id, child in enumerate(self.childrens[img_i]):
                if child.area > self.quadTrees[img_i].minArea + 0.01:
                    ray_num = 10
                else:
                    ray_num = int(child.area * ray_num_per_pixel)       # 最底层叶子节点的光线数由面积*平均每个像素采样点数来决定
                if prob:
                    sharp_images = self.processor.sharp_imgs[img_i]
                    image_block = sharp_images[int(child.x0): int(child.x1), int(child.y0): int(child.y1)]
                    seleted_pixel = self.processor.sample_pixels(image_block, ray_num)
                    interpolated_rgb = self.images[img_i][int(child.x0): int(child.x1), int(child.y0): int(child.y1)
                                       ][seleted_pixel[:, 0], seleted_pixel[:, 1], :].transpose(0, 1)
                    interpolated_dir = self.dirs[img_i][int(child.x0): int(child.x1), int(child.y0): int(child.y1)
                                       ][seleted_pixel[:, 0], seleted_pixel[:, 1], :].transpose(0, 1)
                    interpolated_origin = self.origins[img_i][int(child.x0): int(child.x1), int(child.y0): int(child.y1)
                                          ][seleted_pixel[:, 0], seleted_pixel[:, 1], :].transpose(0, 1)

                    if debug:
                        result_selected_pixel.append(seleted_pixel + torch.Tensor([child.x0, child.y0]))
                else:
                    selected_pixel_x = torch.randint(int(child.x0 * scale), int((child.x1 - 0.01) * scale),
                                                     (ray_num,)) / scale
                    selected_pixel_y = torch.randint(int(child.y0 * scale), int((child.y1 - 0.01) * scale),
                                                     (ray_num,)) / scale
                    seleted_pixel = torch.stack([selected_pixel_x, selected_pixel_y], 1)
                    interpolated_rgb = self.images[img_i][seleted_pixel[:, 0], seleted_pixel[:, 1], :].transpose(0, 1)
                    interpolated_dir = self.dirs[img_i][seleted_pixel[:, 0], seleted_pixel[:, 1], :].transpose(0, 1)
                    interpolated_origin = self.origins[img_i][seleted_pixel[:, 0], seleted_pixel[:, 1], :].transpose(0, 1)

                    if debug:
                        result_selected_pixel.append(seleted_pixel)

                # [tree_ray_num, 2 or 3]

                result_rgb.append(interpolated_rgb)
                result_dir.append(interpolated_dir)
                result_origin.append(interpolated_origin)
                self.result_leaf_id.append(torch.Tensor([[img_i, leaf_id]]).repeat([ray_num, 1]))

            if debug:
                result_selected_pixel = torch.cat(result_selected_pixel, 0)     # [N, 2]
                self.visualize_split_and_sample_points(img_i, result_selected_pixel)
                result_selected_pixel = []

        result_rgb = torch.cat(result_rgb, 1).transpose(0, 1)
        ray_num = result_rgb.shape[0]
        shuffle_index = torch.randperm(ray_num)
        result_rgb = result_rgb[shuffle_index].contiguous()
        result_dir = torch.cat(result_dir, 1).transpose(0, 1)[shuffle_index].contiguous()
        result_origin = torch.cat(result_origin, 1).transpose(0, 1)[shuffle_index].contiguous()
        # [N, 2], N采样像素个数，2分别表示图片id、叶子节点编号
        self.result_leaf_id = torch.cat(self.result_leaf_id, 0)[shuffle_index]

        return Rays(result_origin, result_dir, result_rgb).to(device=self.dataset.device)

    def gen_rays_v3_multiThread(self, down_scale=16, prob=True, rand=0.8, debug=False, last_epoch=False):
        """
        gen_rays_v3的多线程版本
        """
        ray_num_per_image = self.dataset.epoch_size / self.n_images / down_scale
        ray_num_per_pixel = ray_num_per_image / self.h / self.w
        result_rgb = [[] for i in range(self.n_images)]
        result_dir = [[] for i in range(self.n_images)]
        result_origin = [[] for i in range(self.n_images)]
        result_leaf_id = [[] for i in range(self.n_images)]

        thread_vars = []
        # 如果是最后一轮，构造一堆新的、没有分割过的四叉树，因为最后一轮要在全像素点上采样
        if last_epoch:
            ray_num_per_image = self.init_epoch_size / self.n_images / down_scale
            ray_num_per_pixel = ray_num_per_image / self.h / self.w
            new_quadTrees = [QuadTree(self.images[i], 0, 1) for i in range(self.n_images)]
            new_childrens = [get_children(new_quadTrees[i].root) for i in range(self.n_images)]
            for tree_id in range(self.n_images):
                thread_vars.append((None, {
                    'manager': self, 'root': new_quadTrees[tree_id], 'tree_id': tree_id,
                    'children': new_childrens[tree_id], 'ray_num_per_pixel': ray_num_per_pixel,
                    'result_rgb': result_rgb, 'result_dir': result_dir, 'result_origin': result_origin,
                    'result_leaf_id': result_leaf_id, 'prob': prob, 'randSamp_prec': rand, 'debug': debug
                }))
        else:
            for tree_id in range(self.n_images):
                thread_vars.append((None, {
                    'manager': self, 'root': self.quadTrees[tree_id], 'tree_id': tree_id,
                    'children': self.childrens[tree_id], 'ray_num_per_pixel': ray_num_per_pixel,
                    'result_rgb': result_rgb, 'result_dir': result_dir, 'result_origin': result_origin,
                    'result_leaf_id': result_leaf_id, 'prob': prob, 'randSamp_prec': rand, 'debug': debug
                }))

        pool = threadpool.ThreadPool(5)
        # v3非整数点采样，v3_1整数点采样
        # requests = threadpool.makeRequests(gen_rays_v3_subThread, thread_vars)
        requests = threadpool.makeRequests(gen_rays_v3_1_subThread, thread_vars)
        [pool.putRequest(req) for req in requests]
        pool.wait()

        result_rgb = torch.cat(result_rgb, 1).transpose(0, 1)
        result_dir = torch.cat(result_dir, 1).transpose(0, 1)
        result_origin = torch.cat(result_origin, 1).transpose(0, 1)
        result_leaf_id = torch.cat(result_leaf_id, 0)

        shuffle_index = torch.randperm(result_rgb.shape[0])

        result_rgb = result_rgb[shuffle_index].contiguous()
        result_dir = result_dir[shuffle_index].contiguous()
        result_origin = result_origin[shuffle_index].contiguous()
        self.result_leaf_id = result_leaf_id[shuffle_index]

        return Rays(result_origin, result_dir, result_rgb)

    def gen_rays_v4(self, sampler_ret, down_scale=1, debug=False):
        """
        在v4版本里，我们把光线采样做成离线的，在每次调用gen_rays只是去索引每个叶子节点里面采好的颜色和光线。
        """

        ray_num_per_image = self.dataset.epoch_size / self.n_images / down_scale
        ray_num_per_pixel = ray_num_per_image / self.h / self.w
        # 构造光线用
        result_rgb = []
        result_dir = []
        result_origin = []
        # 后面调整四叉树时用
        self.result_leaf_id = []  # [N, 2], N采样像素个数，2分别表示图片id、叶子节点编号
        # debug时，visualize用
        result_selected_pixel = []

        n_block_side = 2 ** (self.cur_level - 1)  # 每个边上有多少段
        block_h_length = self.h / n_block_side
        block_w_length = self.w / n_block_side
        for img_i in tqdm(range(self.n_images)):
            for leaf_id, child in enumerate(self.childrens[img_i]):
                if child.area > self.quadTrees[img_i].minArea + 0.01:
                    ray_num = 10
                    selected_pixel_x = torch.randint(math.ceil(child.x0), math.ceil(child.x1), (ray_num,))
                    selected_pixel_y = torch.randint(math.ceil(child.y0), math.ceil(child.y1 - 0.01), (ray_num,))
                    seleted_pixel = torch.stack([selected_pixel_x, selected_pixel_y], 1)
                    interpolated_rgb = self.images[img_i][seleted_pixel[:, 0], seleted_pixel[:, 1], :]
                    interpolated_dir = self.dirs[img_i][seleted_pixel[:, 0], seleted_pixel[:, 1], :]
                    interpolated_origin = self.origins[img_i][seleted_pixel[:, 0], seleted_pixel[:, 1], :]
                else:
                    all_level_pixel = sampler_ret[img_i][self.cur_level]
                    # all_level_rgb = sampler_ret[img_i][self.cur_level][0]
                    # all_level_dir = sampler_ret[img_i][self.cur_level][1]
                    # all_level_origin = sampler_ret[img_i][self.cur_level][2]
                    ray_num = all_level_pixel.shape[2]

                    coorh = int(child.x0 // block_h_length)
                    coorw = int(child.y0 // block_w_length)
                    block_pixels = all_level_pixel[coorh][coorw]
                    interpolated_rgb = self.images[img_i][block_pixels[:, 0], block_pixels[:, 1], :]
                    interpolated_dir = self.dirs[img_i][block_pixels[:, 0], block_pixels[:, 1], :]
                    interpolated_origin = self.origins[img_i][block_pixels[:, 0], block_pixels[:, 1], :]

                # [tree_ray_num, 2 or 3]
                result_rgb.append(interpolated_rgb)
                result_dir.append(interpolated_dir)
                result_origin.append(interpolated_origin)
                self.result_leaf_id.append(torch.Tensor([[img_i, leaf_id]]).repeat([ray_num, 1]))

            if debug:
                dest_img = 20
                dest_level = 1
                result_selected_pixel = sampler_ret[dest_img][dest_level]
                self.visualize_split_and_sample_points(dest_img, result_selected_pixel)

        result_rgb = torch.cat(result_rgb, 0)
        ray_num = result_rgb.shape[0]
        shuffle_index = torch.randperm(ray_num)
        result_rgb = result_rgb[shuffle_index].contiguous()
        result_dir = torch.cat(result_dir, 0)[shuffle_index].contiguous()
        result_origin = torch.cat(result_origin, 0)[shuffle_index].contiguous()
        # [N, 2], N采样像素个数，2分别表示图片id、叶子节点编号
        self.result_leaf_id = torch.cat(self.result_leaf_id, 0)[shuffle_index]

        return Rays(result_origin, result_dir, result_rgb).to(device=self.dataset.device)


    def adjust_tree(self, rgb_gt, rgb_pred, thres=0.01, debug=False):
        """
        根据每一块叶子节点内的loss，决定是否要对这一块进行细分。如果某一块loss的均值大于给定阈值，就细分，改变self.childrens里这个节点的children属性
        loss跟树和节点编号的对应关系存在self.result_leaf_id中
        # self.result_ray_offset: torch.LongTensor([n_images+1]), 标记每个树上光线数目的偏移量，用于后面分离每个树上的光线
        :param rgb_gt, rgb_pred: [N, 1]
        :return:
        """
        loss = torch.abs(rgb_gt - rgb_pred).detach()

        updated_leaf_num = 0
        for img_i in range(self.n_images):
            img_i_idx = torch.where(self.result_leaf_id[:, 0] == img_i)
            img_i_leaves = self.result_leaf_id[img_i_idx][:, 1]
            img_i_loss = loss[img_i_idx]

            minArea = self.quadTrees[img_i].minArea  # 细分前第i棵树的最小叶子节点面积
            for leaf_i in range(len(self.childrens[img_i])):
                leaf_i_idx = torch.where(img_i_leaves == leaf_i)
                leaf_i_loss = img_i_loss[leaf_i_idx]
                # 如果块内的平均loss大于阈值，进行一次划分
                # 最小loss结果会如何？
                if leaf_i_loss.mean() > thres:
                    assert self.childrens[img_i][leaf_i].children == []
                    # 只有当这个叶子节点是最底层叶子节点时才细分。不然细分的话也没什么用
                    if self.childrens[img_i][leaf_i].area == minArea:
                        self.childrens[img_i][leaf_i].subdivide_once()
                        updated_leaf_num += 1

                        # 更新当前最小叶子节点面积
                        if self.quadTrees[img_i].minArea == minArea:
                            self.quadTrees[img_i].minArea /= 4

        print('{} leaves subdivided.'.format(updated_leaf_num))
        # 更新self.childrens
        self.childrens = [get_children(self.quadTrees[i].root) for i in range(self.n_images)]
        self.cur_level += 1

        if debug:
            self.visualize_subdivide(tree_id=-1)

    def adjust_tree_multiThread(self, rgb_gt, rgb_pred, thres=0.01, debug=False):
        """
        self.adjust_tree函数的多线程方法
        :return:
        """
        loss = torch.abs(rgb_gt - rgb_pred).detach()

        if debug:
            # TODO
            self.visualize_image_split_and_mean_loss(18, loss)

        thread_vars = []
        for tree_id in range(self.n_images):
            thread_vars.append((None, {
                'manager': self, 'img_i': tree_id, 'result_leaf_id': self.result_leaf_id, 'loss': loss,
                'children': self.childrens[tree_id], 'thres': thres, 'tree_i': self.quadTrees[tree_id]
            }))

        pool = threadpool.ThreadPool(5)
        requests = threadpool.makeRequests(adjust_tree_subThread, thread_vars)
        [pool.putRequest(req) for req in requests]
        pool.wait()

        total_child_num = sum([len(children) for children in self.childrens])
        print('After sudivide, there are {} child nodes'.format(total_child_num))
        self.cur_level += 1

        if False:
            self.visualize_subdivide(tree_id=-1)


    def visualize_loss_distribute(self, loss):
        """
        在图片上给每个像素点的loss染色，loss越低，越从红色变为绿色。这是为了展示随着训练的进行，空白区域确实不需要投入太多关注
        :param loss:
        :return:
        """
        pass


def gen_rays_v3_subThread(manager: QuadTreeManager, root: QuadTree, tree_id: int, children, ray_num_per_pixel: int,
                          result_rgb: List, result_dir: List, result_origin: List,
                          result_leaf_id: List):
    """
    这是QuadTreeManager.gen_rays_v3的多线程版本。
    优化了一版：之前是对每个叶子节点分别采样像素->插值，最后拼接；现改为分别采样像素后拼接，统一插值
    """
    scale = 1000
    all_normalized_pixel = []
    for leaf_id, child in enumerate(children):
        if child.area > root.minArea + 0.01:
            ray_num = 10
        else:
            ray_num = int(child.area * ray_num_per_pixel)  # 最底层叶子节点的光线数由面积*平均每个像素采样点数来决定
        selected_pixel_x = torch.randint(int((child.x0 + 0.001) * scale), int((child.x1 - 0.001) * scale),
                                         (ray_num,)) / scale
        selected_pixel_y = torch.randint(int((child.y0 + 0.001) * scale), int((child.y1 - 0.001) * scale),
                                         (ray_num,)) / scale
        seleted_pixel = torch.stack([selected_pixel_x, selected_pixel_y], 1)

        # 归一化到[-1, 1]
        normalized_pixel = seleted_pixel / torch.Tensor([manager.h / 2, manager.w / 2]) - 1
        all_normalized_pixel.append(normalized_pixel)

        result_leaf_id[tree_id].append(torch.Tensor([[tree_id, leaf_id]]).repeat([ray_num, 1]))

    # [1, tree_ray_num, 1, 2]
    all_normalized_pixel = torch.cat(all_normalized_pixel, 0).unsqueeze(0).unsqueeze(2)

    # debug，看一下采样点的分布情况
    if False:
        paint_pixel = (all_normalized_pixel.squeeze() + 1) * torch.Tensor([manager.h / 2, manager.w / 2])
        manager.visualize_split_and_sample_points(tree_id, paint_pixel)

    # [2 or 3, tree_ray_num]
    interpolated_rgb = F.grid_sample(manager.images[tree_id: tree_id+1].permute(0, 3, 1, 2),
                                     all_normalized_pixel).squeeze()
    interpolated_dir = F.grid_sample(manager.dirs[tree_id: tree_id+1].permute(0, 3, 1, 2),
                                     all_normalized_pixel).squeeze()
    interpolated_origin = F.grid_sample(manager.origins[tree_id: tree_id+1].permute(0, 3, 1, 2),
                                        all_normalized_pixel).squeeze()

    result_rgb[tree_id] = interpolated_rgb
    result_dir[tree_id] = interpolated_dir
    result_origin[tree_id]= interpolated_origin
    result_leaf_id[tree_id] = torch.cat(result_leaf_id[tree_id], 0)

    # print('gen rays: subthread{} completed.'.format(tree_id))

def gen_rays_v3_1_subThread(manager: QuadTreeManager, root: QuadTree, tree_id: int, children, ray_num_per_pixel: int,
                          result_rgb: List, result_dir: List, result_origin: List,
                          result_leaf_id: List, prob: bool, randSamp_prec: float, debug: bool):
    """
    这是QuadTreeManager.gen_rays_v3_subThread的3.1版本，在这里我们只用整数像素点采样光线，目的是证明非整数像素点采光线是有效的
    """

    all_selected_pixel = []
    for leaf_id, child in enumerate(children):
        if child.area > root.minArea + 0.01:
            ray_num = 10
        else:
            ray_num = int(child.area * ray_num_per_pixel)  # 最底层叶子节点的光线数由面积*平均每个像素采样点数来决定

        if prob:
            # sharp_images = manager.processor.sharp_imgs[tree_id]
            # image_block = sharp_images[int(child.x0): int(child.x1), int(child.y0): int(child.y1)]
            # seleted_pixel = manager.processor.sample_pixels(image_block, ray_num)
            # all_selected_pixel.append(seleted_pixel + torch.LongTensor([int(child.x0), int(child.y0)]))

            # 一大半光线用概率方法采，一小半光线随机采
            ray_num1 = int(ray_num*(1-randSamp_prec))
            ray_num2 = ray_num - ray_num1
            sharp_images = manager.processor.sharp_imgs[tree_id]
            image_block = sharp_images[int(child.x0): int(child.x1), int(child.y0): int(child.y1)]
            seleted_pixel = manager.processor.sample_pixels(image_block, ray_num1)
            all_selected_pixel.append(seleted_pixel + torch.LongTensor([int(child.x0), int(child.y0)]))

            selected_pixel_x = torch.randint(math.ceil(child.x0), math.ceil(child.x1), (ray_num2,))
            selected_pixel_y = torch.randint(math.ceil(child.y0), math.ceil(child.y1 - 0.01), (ray_num2,))
            seleted_pixel = torch.stack([selected_pixel_x, selected_pixel_y], 1)
            all_selected_pixel.append(seleted_pixel)

        else:
            selected_pixel_x = torch.randint(math.ceil(child.x0), math.ceil(child.x1), (ray_num,))
            selected_pixel_y = torch.randint(math.ceil(child.y0), math.ceil(child.y1 - 0.01), (ray_num,))
            seleted_pixel = torch.stack([selected_pixel_x, selected_pixel_y], 1)
            all_selected_pixel.append(seleted_pixel)

        result_leaf_id[tree_id].append(torch.Tensor([[tree_id, leaf_id]]).repeat([ray_num, 1]))

    # [tree_ray_num, 2]
    all_normalized_pixel = torch.cat(all_selected_pixel, 0)

    # debug，看一下采样点的分布情况
    if debug:
        # TODO
        if tree_id == 18:
            manager.visualize_split_and_sample_points(tree_id, all_normalized_pixel)

    # [tree_ray_num, 2 or 3]
    interpolated_rgb = manager.images[tree_id][all_normalized_pixel[:, 0], all_normalized_pixel[:, 1], :].transpose(0, 1)
    interpolated_dir = manager.dirs[tree_id][all_normalized_pixel[:, 0], all_normalized_pixel[:, 1], :].transpose(0, 1)
    interpolated_origin = manager.origins[tree_id][all_normalized_pixel[:, 0], all_normalized_pixel[:, 1], :].transpose(0, 1)

    result_rgb[tree_id] = interpolated_rgb
    result_dir[tree_id] = interpolated_dir
    result_origin[tree_id]= interpolated_origin
    result_leaf_id[tree_id] = torch.cat(result_leaf_id[tree_id], 0)

    # print('gen rays: subthread{} completed.'.format(tree_id))


def adjust_tree_subThread(manager: QuadTreeManager, img_i: int, result_leaf_id: torch.Tensor, loss: torch.Tensor,
                          children: List, thres: float, tree_i: QuadTree):

    img_i_idx = torch.where(result_leaf_id[:, 0] == img_i)
    img_i_leaves = result_leaf_id[img_i_idx][:, 1]
    img_i_loss = loss[img_i_idx]
    minArea = tree_i.minArea    # 细分前第i棵树的最小叶子节点面积
    for leaf_i in range(len(children)):
        leaf_i_idx = torch.where(img_i_leaves == leaf_i)
        leaf_i_loss = img_i_loss[leaf_i_idx]
        # 如果块内的平均loss大于阈值，进行一次划分
        # 最小loss结果会如何？
        if leaf_i_loss.mean() > thres:
            assert children[leaf_i].children == []
            # 只有当这个叶子节点是最底层叶子节点时才细分。不然细分的话也没什么用
            if children[leaf_i].area == minArea:
                children[leaf_i].subdivide_once()

                # 更新当前最小叶子节点面积
                if tree_i.minArea == minArea:
                    tree_i.minArea /= 4

    manager.childrens[img_i] = get_children(tree_i.root)


def recursive_subdivide(node: QuadTreeNode, thres: float, image: torch.Tensor, cur_depth, max_depth):
    if cur_depth >= max_depth:
        return
    if node.get_error(image) < thres:
        return

    midx = (node.x0+node.x1)/2
    midy = (node.y0+node.y1)/2
    # 左上
    node1 = QuadTreeNode(node.x0, node.y0, midx, midy)
    recursive_subdivide(node1, thres, image, cur_depth+1, max_depth)
    # 左下
    node2 = QuadTreeNode(midx, node.y0, node.x1, midy)
    recursive_subdivide(node2, thres, image, cur_depth+1, max_depth)
    # 右上
    node3 = QuadTreeNode(node.x0, midy, midx, node.y1)
    recursive_subdivide(node3, thres, image, cur_depth+1, max_depth)
    # 右下
    node4 = QuadTreeNode(midx, midy, node.x1, node.y1)
    recursive_subdivide(node4, thres, image, cur_depth+1, max_depth)

    node.children = [node1, node2, node3, node4]


def get_children(node: QuadTreeNode):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (get_children(child))
    return children

