import torch
from cv2 import cv2
import numpy as np

def get_img_prob(img):
    """
    获得逐像素点的临近像素方差的灰度值
    :param img: [h, w, 3]
    :return: [h, w]
    """
    raw_shape = img.shape[:2]
    kernel = (3,3)
    img = np.array(img, dtype=np.float64)
    E_square = cv2.blur(img ** 2, kernel)
    square_E = cv2.blur(img, kernel) ** 2
    # sharp_img = cv2.sqrt(np.abs(E_square - square_E))
    sharp_img = np.abs(E_square - square_E)
    gray_img = sharp_img.sum(2)     # [h, w]
    # if True:
    #     cv2.imwrite('debug/variation_pixel.jpg', gray_img)

    gray_img = np.asarray(gray_img, dtype=np.float64)
    gray_img = gray_img.flatten() + 1e-6
    # gray_min = 0.01 * np.mean(gray_img)  # 防止0的位置被归一化为0，这是一个可调节的阈值
    # gray_min = np.median(gray_img)
    # gray_max = np.max(gray_img)
    # gray_img = np.clip(gray_img, gray_min, gray_max)
    # gray_img = (1 / (1 + np.exp(-gray_img)) - 0.5) * 0.6 + 0.5  # sigmoid [0,1]->[0,0.8]
    # gray_img = 1 / (1 + np.exp(-gray_img))                  # sigmoid [0,1]
    # gray_img = (1 / (1 + np.exp(-gray_img)) - 0.5) * 2 + 0.5  # sigmoid [0,1]->[0,1.5]
    # gray_img = (1 / (1 + np.exp(-gray_img)) - 0.5) * 3 + 0.5  # sigmoid [0,1]->[0,2]
    # gray_img = (1 / (1 + np.exp(-gray_img)) - 0.5) * 5 + 0.5  # sigmoid [0,1]->[0,3]
    # gray_img = (1 / (1 + np.exp(-gray_img)) - 0.5) * 7 + 0.5  # sigmoid [0,1]->[0,4]    # 在fox数据集上这个是最好的
    # gray_img = (1 / (1 + np.exp(-gray_img)) - 0.5) * 9 + 0.5  # sigmoid [0,1]->[0,5]
    gray_max = np.max(gray_img)

    # debug: TIP revision可视化用
    # gray_img = (1 / (1 + np.exp(-0.1*gray_img)) - 0.5) * 3 + 0.5
    # cv2.imwrite('debug/variation_pixel.jpg', np.asarray(gray_img.reshape(raw_shape)*255/2, dtype=np.uint8))
    # debug：用熵来计算
    entropy = cal_entropy(img)
    entropy = entropy.max() - entropy
    entropy = entropy * (255/entropy.max())
    cv2.imwrite('debug/variation_pixel.jpg', np.asarray(entropy, dtype=np.uint8))

    # gray_img = (gray_img - 0) / (gray_max - 0)
    gray_softmax = gray_img / (np.sum(gray_img))
    gray_softmax = np.reshape(gray_softmax, raw_shape)
    return gray_softmax


def cal_entropy(img):
    h, w, _ = img.shape
    idx = torch.arange(h*w)
    # 3*3
    idx0 = idx - w - 1
    idx1 = idx - w
    idx2 = idx - w + 1
    idx3 = idx - 1
    idx4 = idx + 1
    idx5 = idx + w - 1
    idx6 = idx + w
    idx7 = idx + w + 1
    neighbor_idx = torch.stack([idx0, idx1, idx2, idx3, idx, idx4, idx5, idx6, idx7], 1)  # [h*w, 9]
    border_mask = (idx % w == 0) | (idx % w == (w - 1)) | (idx // w == 0) | (idx // w == (h - 1))
    neighbor_idx[border_mask] = idx[border_mask].unsqueeze(1).repeat([1, 9])

    # 5*5
    # idx0 = idx - w - 2
    # idx1 = idx - w
    # idx2 = idx - w + 2
    # idx3 = idx - 2
    # idx4 = idx + 2
    # idx5 = idx + w - 2
    # idx6 = idx + w
    # idx7 = idx + w + 2
    # neighbor_idx = torch.stack([idx0, idx1, idx2, idx3, idx, idx4, idx5, idx6, idx7], 1)  # [h*w, 9]
    # border_mask = (idx % w == 0) | (idx % w == 1) | (idx % w == (w - 1)) |(idx % w == (w - 2)) |\
    #               (idx // w == 0) | (idx // w == 1) | (idx // w == (h - 1)) | (idx // w == (h - 2))
    # neighbor_idx[border_mask] = idx[border_mask].unsqueeze(1).repeat([1, 9])

    # 7*7
    # idx0 = idx - w - 3
    # idx1 = idx - w
    # idx2 = idx - w + 3
    # idx3 = idx - 3
    # idx4 = idx + 3
    # idx5 = idx + w - 3
    # idx6 = idx + w
    # idx7 = idx + w + 3
    # neighbor_idx = torch.stack([idx0, idx1, idx2, idx3, idx, idx4, idx5, idx6, idx7], 1)  # [h*w, 9]
    # border_mask = (idx % w == 0) | (idx % w == 1) | (idx % w == 2) | (idx % w == (w - 1)) | (idx % w == (w - 2)) | (idx % w == (w - 3)) |  \
    #               (idx // w == 0) | (idx // w == 1) | (idx // w == 2) | (idx // w == (h - 1)) | (idx // w == (h - 2)) | (idx // w == (h - 3))
    # neighbor_idx[border_mask] = idx[border_mask].unsqueeze(1).repeat([1, 9])

    # 9*9
    # idx0 = idx - w - 4
    # idx1 = idx - w
    # idx2 = idx - w + 4
    # idx3 = idx - 4
    # idx4 = idx + 4
    # idx5 = idx + w - 4
    # idx6 = idx + w
    # idx7 = idx + w + 4
    # neighbor_idx = torch.stack([idx0, idx1, idx2, idx3, idx, idx4, idx5, idx6, idx7], 1)  # [h*w, 9]
    # border_mask = (idx % w == 0) | (idx % w == 1) | (idx % w == 2) | (idx % w == 3) | (idx % w == (w - 1)) | (idx % w == (w - 2)) | (
    #             idx % w == (w - 3)) | (idx % w == (w - 4)) | \
    #               (idx // w == 0) | (idx // w == 1) | (idx // w == 2) | (idx // w == 3) | (idx // w == (h - 1)) | (
    #                           idx // w == (h - 2)) | (idx // w == (h - 3)) | (idx // w == (h - 4))
    # neighbor_idx[border_mask] = idx[border_mask].unsqueeze(1).repeat([1, 9])

    img = torch.Tensor(img).reshape(-1, 3)
    neighbor_pixel = img[neighbor_idx]      # [h*w, 9, 3]
    neighbor_pixel = torch.softmax(neighbor_pixel, 1)
    entropy_rgb = (-neighbor_pixel*torch.log(neighbor_pixel+1e-8)).sum(1)       #[h*w, 3]
    entropy = entropy_rgb.sum(1)
    return entropy.numpy().reshape(h, w)


def sample_pixels(image_prob, sample_num=4096):
    """
    从图片中采样sample_num个采样点，每个像素点的采样概率由它跟它周围像素的方差决定
    :param image: [h, w],该image是self.get_sharp_img已经计算好方差的灰度值图片
    :param sample_num: int
    :return: torch.LongTensor([sample_num, 2])
    """
    n, h, w = image_prob.shape
    sample_index = np.random.choice(h*w, sample_num, p=image_prob.reshape(-1), replace=True)

    sample_x = np.floor(sample_index / w)       # 所有采样点的行坐标
    sample_y = sample_index - sample_x * w      # 所有采样点的列坐标
    sample_x = torch.LongTensor(sample_x)
    sample_y = torch.LongTensor(sample_y)
    selected_pixel = torch.stack([sample_x, sample_y], 1)

    if False:
        visualize_sample_points(selected_pixel)

    return sample_index

def sample_pixels_torch(image_prob, sample_num=4096):
    """
    上面sample_pixels的pytorch cuda版本（为了节省numpy和torch之间转换的时间）
    :param image_prob:
    :param sample_num:
    :return:
    """
    # p = image_prob.reshape(-1)
    # sample_index = p.multinomial(num_samples=sample_num, replacement=False)
    sample_index = image_prob.multinomial(num_samples=sample_num, replacement=True)  # replacement=True比False快了大约7倍！

    return sample_index


def visualize_object_and_sample_points(img, selected_pixel):
    """
    可视化四叉树在树上的采样点，连带着物体
    """
    imgc = self.images[img_id].numpy().copy() * 255.0

    for x, y in selected_pixel.numpy():
        imgc = cv2.circle(imgc, (int(y), int(x)), 1, (255, 0, 0), -1)

    cv2.imwrite('debug/ImageProcessor_object_and_sample_points_{}.jpg'.format(img_id), imgc[:, :, [2, 1, 0]])

def visualize_sample_points(selected_pixel):
    """
    可视化四叉树在树上的采样点（只可视化采样点，背景是白色）
    """
    imgc = np.ones([1920, 1080, 3]) * 255

    for x, y in selected_pixel.numpy():
        imgc = cv2.circle(imgc, (int(y), int(x)), 1, (255, 0, 0), -1)

    cv2.imwrite('debug/ImageProcessor_sample_points.jpg', imgc[:, :, [2, 1, 0]])

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


if __name__ == '__main__':
    # 测试一下replacement=True or False的速度差异
    import time
    p = torch.rand(800*800).cuda()
    p = p / p.sum()
    t0 = time.time()
    for i in range(1000):
        ind = torch.multinomial(p, 4096, replacement=False)
    print(time.time() - t0)
    t0 = time.time()
    for i in range(1000):
        ind = torch.multinomial(p, 4096, replacement=True)
    print(time.time() - t0)
    t0 = time.time()
    for i in range(1000):
        ind = torch.randint(0, 800*800, size=[4096])
    print(time.time() - t0)
    p = torch.ones(800*800).cuda()
    p = p / p.sum()
    t0 = time.time()
    for i in range(1000):
        ind = torch.multinomial(p, 4096, replacement=False)
    print(time.time() - t0)