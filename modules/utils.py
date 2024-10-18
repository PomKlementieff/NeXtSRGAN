import cv2
import yaml
import sys
import time
import numpy as np
import tensorflow as tf
from absl import logging
from modules.dataset import load_tfrecord_dataset


def load_yaml(load_path):
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    return loaded


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            logging.info(e)


def load_dataset(cfg, key, shuffle=True, buffer_size=10240):
    dataset_cfg = cfg[key]
    logging.info("load {} from {}".format(key, dataset_cfg['path']))
    dataset = load_tfrecord_dataset(
        tfrecord_name=dataset_cfg['path'],
        batch_size=cfg['batch_size'],
        gt_size=cfg['gt_size'],
        scale=cfg['scale'],
        shuffle=shuffle,
        using_bin=dataset_cfg['using_bin'],
        using_flip=dataset_cfg['using_flip'],
        using_rot=dataset_cfg['using_rot'],
        buffer_size=buffer_size)
    return dataset


def create_lr_hr_pair(raw_img, scale=4.):
    lr_h, lr_w = raw_img.shape[0] // scale, raw_img.shape[1] // scale
    hr_h, hr_w = lr_h * scale, lr_w * scale
    hr_img = raw_img[:hr_h, :hr_w, :]
    lr_img = imresize_np(hr_img, 1 / scale)
    return lr_img, hr_img


def tensor2img(tensor):
    return (np.squeeze(tensor.numpy()).clip(0, 1) * 255).astype(np.uint8)


def change_weight(model, vars1, vars2, alpha=1.0):
    for i, var in enumerate(model.trainable_variables):
        var.assign((1 - alpha) * vars1[i] + alpha * vars2[i])


class ProgressBar(object):
    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            logging.info('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        self.completed = 0

    def update(self, inf_str=''):
        self.completed += 1
        if not self.warm_up:
            self.start_time = time.time() - 1e-2
            self.warm_up = True
        elapsed = time.time() - self.start_time
        fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec'
        sys.stdout.write(stdout_str.format(
            bar_chars, self.completed, self.task_num, inf_str, fps))
        sys.stdout.flush()


def imresize_np(img, scale, antialiasing=True):
    in_H, in_W, in_C = img.shape
    _, out_H, out_W = in_C, np.ceil(in_H * scale), np.ceil(in_W * scale)
    out_H, out_W = out_H.astype(np.int64), out_W.astype(np.int64)
    kernel_width = 4
    kernel = 'cubic'
    
    weights_H, indices_H, sym_len_Hs, sym_len_He = _calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = _calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    
    img_aug = np.zeros(((in_H + sym_len_Hs + sym_len_He), in_W, in_C))
    img_aug[sym_len_Hs:sym_len_Hs + in_H] = img

    sym_patch = img[:sym_len_Hs, :, :]
    sym_patch_inv = sym_patch[::-1]
    img_aug[0:sym_len_Hs] = sym_patch_inv

    sym_patch = img[-sym_len_He:, :, :]
    sym_patch_inv = sym_patch[::-1]
    img_aug[sym_len_Hs + in_H:sym_len_Hs + in_H + sym_len_He] = sym_patch_inv

    out_1 = np.zeros((out_H, in_W, in_C))
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = weights_H[i].dot(
            img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1))
        out_1[i, :, 1] = weights_H[i].dot(
            img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1))
        out_1[i, :, 2] = weights_H[i].dot(
            img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1))

    out_1_aug = np.zeros((out_H, in_W + sym_len_Ws + sym_len_We, in_C))
    out_1_aug[:, sym_len_Ws:sym_len_Ws + in_W] = out_1

    sym_patch = out_1[:, :sym_len_Ws, :]
    sym_patch_inv = sym_patch[:, ::-1]
    out_1_aug[:, 0:sym_len_Ws] = sym_patch_inv

    sym_patch = out_1[:, -sym_len_We:, :]
    sym_patch_inv = sym_patch[:, ::-1]
    out_1_aug[:, sym_len_Ws + in_W:sym_len_Ws + in_W + sym_len_We] = \
        sym_patch_inv

    out_2 = np.zeros((out_H, out_W, in_C))
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].dot(
            weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].dot(
            weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].dot(
            weights_W[i])
        
    return out_2.clip(0, 255)


def _cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).astype(np.float64)) \
        + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
            ((absx > 1) * (absx <= 2)).astype(np.float64))


def _calculate_weights_indices(in_length, out_length, scale, kernel,
                               kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        kernel_width = kernel_width / scale

    x = np.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)

    P = (np.ceil(kernel_width) + 2).astype(np.int32)

    indices = left.reshape(int(out_length), 1).repeat(P, axis=1) + \
        np.linspace(0, P - 1, P).reshape(1, int(P)).repeat(out_length, axis=0)

    distance_to_center = u.reshape(int(out_length), 1).repeat(P, axis=1) - indices

    if (scale < 1) and (antialiasing):
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    weights_sum = np.sum(weights, 1).reshape(int(out_length), 1)
    weights = weights / weights_sum.repeat(P, axis=1)

    weights_zero_tmp = np.sum((weights == 0), 0)
    if not np.isclose(weights_zero_tmp[0], 0, rtol=1e-6):
        indices = indices[:, 1:1 + int(P) - 2]
        weights = weights[:, 1:1 + int(P) - 2]
    if not np.isclose(weights_zero_tmp[-1], 0, rtol=1e-6):
        indices = indices[:, 0:0 + int(P) - 2]
        weights = weights[:, 0:0 + int(P) - 2]
    weights = weights.copy()
    indices = indices.copy()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) \
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    
    # BGR to RGB
    img = img[:, :, ::-1]

    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img, [[24.966, 112.0, -18.214],
                  [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    """Convert BGR image to YCbCr image."""
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img, [[24.966, 112.0, -18.214],
                  [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    """Convert YCbCr image to RGB image."""
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    
    rlt = np.matmul(
        img, [[0.00456621, 0.00456621, 0.00456621],
              [0, -0.00153632, 0.00791071],
              [0.00625893, -0.00318811, 0]]) * 255.0 + \
        [-222.921, 135.576, -276.836]
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    """Mod crop images by scale factor."""
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def shave(img_in, border=0):
    """Shave border pixels from image."""
    img = np.copy(img_in)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img


def channel_convert(in_c, tar_type, img_list):
    """Convert channel number and type of image list."""
    # conversion_type = in_c + '_' + tar_type
    if in_c == 3 and tar_type == 'y':
        # remove the following two lines when using Y channel only
        img_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in img_list]
    elif in_c == 1 and tar_type == 'RGB':
        return [np.repeat(img, 3, axis=2) for img in img_list]
    else:
        return img_list


def augment(img_list, hflip=True, rot=True):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees)."""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert tensor to image numpy array."""
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    """Save image."""
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(img1, img2):
    """Calculate SSIM (Structural Similarity)."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """Calculate SSIM (Structural Similarity)."""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')