import torch
from colour import Color

from cv2 import cv2
import numpy as np
from tqdm import tqdm

from util.util import Rays

class ImageProcessor:
    def __init__(self, dataset, scale=50):
        """

        :param dataset:
        :param scale: 灰度图片缩放比例，把灰度值统一除以这个比例，防止不同区域概率差异太大
        """
        self.scale = scale
        self.n_images, self.h, self.w = dataset.n_images, dataset.h_full, dataset.w_full
        rays = dataset.rays
        self.dataset = dataset
        self.init_epoch_size = dataset.epoch_size
        self.images = rays.gt.reshape(self.n_images, self.h, self.w, -1)
        self.dirs = rays.dirs.reshape(self.n_images, self.h, self.w, -1)
        self.origins = rays.origins.reshape(self.n_images, self.h, self.w, -1)
        self.device = self.images.device

        self.images_np = self.images.cpu().detach().numpy() * 255
        self.sharp_imgs = [self.get_sharp_img(self.images_np[i]) for i in range(self.n_images)]

    def get_sharp_img(self, img):
        """
        获得逐像素点的临近像素方差的灰度值
        :param img: [h, w, 3]
        :return: [h, w]
        """
        kernel = (3, 3)
        E_square = cv2.blur(img ** 2, kernel)
        square_E = cv2.blur(img, kernel) ** 2
        sharp_img = cv2.sqrt(np.abs(E_square - square_E))
        gray = cv2.cvtColor(sharp_img[:, :, [2, 1, 0]], cv2.COLOR_BGR2GRAY)
        if False:
            cv2.imwrite('debug/variation_pixel.jpg', gray)
        return gray

    def to_prob(self, gray_img):
        """
        计算给定图片的逐像素点采样概率
        :param gray_img: [h, w, 3]
        :return: [h, w]
        """

        raw_shape = gray_img.shape
        gray_img = np.asarray(gray_img, dtype=np.float64)

        if self.scale == -1:
            gray_img = np.ones_like(gray_img.flatten())
        else:
            gray_img = gray_img.flatten() / self.scale
        gray_softmax = np.exp(gray_img) / np.sum(np.exp(gray_img))
        gray_softmax = np.reshape(gray_softmax, raw_shape)

        return gray_softmax

    def to_prob_v2(self, gray_img):
        # 灰度值统一归一化为[0, 1]后过softmax，作为采样概率
        raw_shape = gray_img.shape
        gray_img = np.asarray(gray_img, dtype=np.float64)

        gray_img = gray_img.flatten() + 1e-6
        gray_min = 0.01 * np.mean(gray_img)    # 防止0的位置被归一化为0，这是一个可调节的阈值
        gray_max = np.max(gray_img)
        gray_img = np.clip(gray_img, gray_min, gray_max)
        gray_img = (gray_img - 0) / (gray_max - 0)

        # cv2.imwrite('debug/variation_pixel.jpg', np.reshape(gray_img, raw_shape) * 255)

        gray_softmax = gray_img / (np.sum(gray_img))
        gray_softmax = np.reshape(gray_softmax, raw_shape)
        return gray_softmax

    def sample_pixels(self, image, sample_num=320000):
        """
        从图片中采样sample_num个采样点，每个像素点的采样概率由它跟它周围像素的方差决定
        :param image: [h, w],该image是self.get_sharp_img已经计算好方差的灰度值图片
        :param sample_num: int
        :return: torch.LongTensor([sample_num, 2])
        """
        prob = self.to_prob_v2(image)
        if False:
            self.visualize_prob_distribution(prob)
        h, w = prob.shape
        sample_index = np.random.choice(h*w, sample_num, p=prob.reshape(-1))

        sample_x = np.floor(sample_index / w)       # 所有采样点的行坐标
        sample_y = sample_index - sample_x * w      # 所有采样点的列坐标
        sample_x = torch.LongTensor(sample_x)
        sample_y = torch.LongTensor(sample_y)
        seleted_pixel = torch.stack([sample_x, sample_y], 1)

        return seleted_pixel

    def visualize_object_and_sample_points(self, img_id, selected_pixel):
        """
        可视化四叉树在树上的采样点，连带着物体
        """
        imgc = self.images[img_id].numpy().copy() * 255.0

        for x, y in selected_pixel.numpy():
            imgc = cv2.circle(imgc, (int(y), int(x)), 1, (255, 0, 0), -1)

        cv2.imwrite('debug/ImageProcessor_object_and_sample_points_{}.jpg'.format(img_id), imgc[:, :, [2, 1, 0]])

    def visualize_sample_points(self, img_id, selected_pixel):
        """
        可视化四叉树在树上的采样点（只可视化采样点，背景是白色）
        """
        imgc = np.ones([self.h, self.w, 3]) * 255

        for x, y in selected_pixel.numpy():
            imgc = cv2.circle(imgc, (int(y), int(x)), 1, (255, 0, 0), -1)

        cv2.imwrite('debug/ImageProcessor_sample_points_{}.jpg'.format(img_id), imgc[:, :, [2, 1, 0]])

    def visualize_sample_point_colors(self, img_id, selected_pixel):
        """
        可视化四叉树在树上的采样点对应的图片颜色，检查代码是否有错误
        """
        imgc = np.zeros([self.h, self.w, 3])

        for x, y in selected_pixel.numpy():
            rgb = self.images[img_id][x][y] * 255
            imgc = cv2.circle(imgc, (int(y), int(x)), 1, (int(rgb[0]), int(rgb[1]), int(rgb[2])), -1)

        cv2.imwrite('debug/ImageProcessor_sample_point_colors_{}.jpg'.format(img_id), imgc[:, :, [2, 1, 0]])

    def visualize_prob_distribution(self, prob_img):
        """
        把不同位置的采样概率分布在图片上显示出来，越红色表示概率越大，越绿色越小
        :param prob_img:
        :return:
        """
        h, w = prob_img.shape
        imgc = np.zeros([self.h, self.w, 3])
        # startColor = Color('red')
        # endColor = Color('green')
        # colorGradient = list(startColor.range_to(endColor, h*w))
        # prob_sorted = np.sort(prob_img.flatten())
        # index_sorted = np.argsort(prob_img.flatten())
        # for idx, idx_sorted in enumerate(index_sorted):
        #     rgb = colorGradient[idx].rgb
        #     rgb = [int(255 * rgb[i]) for i in range(3)]
        #     i = np.floor(idx_sorted[idx] / w)
        #     j = idx_sorted[idx] - i * w
        #     imgc = cv2.circle(imgc, (int(j), int(i)), 1, (rgb[0], rgb[1], rgb[2]), -1)
        min_distance = np.min(prob_img)
        median_distance = np.mean(prob_img)*2
        max_distance = np.max(prob_img)
        for i in range(h):
            for j in range(w):
                dist = prob_img[i][j]
                if dist <= median_distance:
                    r = int(255 * (dist - min_distance) / (median_distance - min_distance+1e-6))
                    g = 255
                    b = 0
                else:
                    r = 255
                    g = int(255 - 255 * (dist - median_distance) / (max_distance - median_distance+1e-6))
                    b = 0
                imgc = cv2.circle(imgc, (j, i), 1, (r, g, b), -1)
        cv2.imwrite('debug/ImageProcessor_prob_distribution.jpg', imgc[:, :, [2, 1, 0]])

    def gen_rays(self, down_scale=2, debug=False):
        ray_num_per_image = int(self.dataset.epoch_size / down_scale / self.n_images)

        result_rgb = []
        result_dir = []
        result_origin = []
        for img_i in tqdm(range(self.n_images)):
            gray_img = self.get_sharp_img(self.images[img_i])
            selected_pixel = self.sample_pixels(gray_img, ray_num_per_image)
            result_rgb.append(self.images[img_i][selected_pixel[:, 0], selected_pixel[:, 1], :])
            result_dir.append(self.dirs[img_i][selected_pixel[:, 0], selected_pixel[:, 1], :])
            result_origin.append(self.origins[img_i][selected_pixel[:, 0], selected_pixel[:, 1], :])

            if debug:
                self.visualize_sample_points(img_i, selected_pixel)

        result_rgb = torch.cat(result_rgb, 0)
        ray_num = result_rgb.shape[0]
        shuffle_index = torch.randperm(ray_num)
        result_rgb = result_rgb[shuffle_index].contiguous()
        result_dir = torch.cat(result_dir, 0)[shuffle_index].contiguous()
        result_origin = torch.cat(result_origin, 0)[shuffle_index].contiguous()

        return Rays(result_origin, result_dir, result_rgb).to(device=self.dataset.device)