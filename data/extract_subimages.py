import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
import time
from shutil import get_terminal_size

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

def main():
    mode = 'pair'
    opt = {
        'n_thread': 20,
        'compression_level': 3
    }

    if mode == 'single':
        opt.update({
            'input_folder': './data/DIV2K/DIV2K_train_HR',
            'save_folder': './data/DIV2K/DIV2K800_sub',
            'crop_sz': 480,
            'step': 240,
            'thres_sz': 48
        })
        extract_single(opt)
    elif mode == 'pair':
        GT_folder = './data/KID_F/KID_F_train_HR'
        LR_folder = './data/KID_F/KID_F_train_LR_bicubic/X4'
        save_GT_folder = './data/KID_F/KID_F_800_sub'
        save_LR_folder = './data/KID_F/KID_F_800_sub_bicLRx4'
        scale_ratio = 4
        crop_sz = 480
        step = 240
        thres_sz = 48

        img_GT_list = _get_paths_from_images(GT_folder)
        img_LR_list = _get_paths_from_images(LR_folder)
        assert len(img_GT_list) == len(img_LR_list), 'Different length of GT_folder and LR_folder.'
        for path_GT, path_LR in zip(img_GT_list, img_LR_list):
            img_GT = Image.open(path_GT)
            img_LR = Image.open(path_LR)
            w_GT, h_GT = img_GT.size
            w_LR, h_LR = img_LR.size
            assert w_GT / w_LR == scale_ratio, f'GT width [{w_GT}] is not {scale_ratio}X as LR width [{w_LR}] for {path_GT}.'
            assert h_GT / h_LR == scale_ratio, f'GT height [{h_GT}] is not {scale_ratio}X as LR height [{h_LR}] for {path_GT}.'
   
        assert crop_sz % scale_ratio == 0, f'crop size is not {scale_ratio}X multiplication.'
        assert step % scale_ratio == 0, f'step is not {scale_ratio}X multiplication.'
        assert thres_sz % scale_ratio == 0, f'thres_sz is not {scale_ratio}X multiplication.'
        
        print('Process GT...')
        opt.update({
            'input_folder': GT_folder,
            'save_folder': save_GT_folder,
            'crop_sz': crop_sz,
            'step': step,
            'thres_sz': thres_sz
        })
        extract_single(opt)
        
        print('Process LR...')
        opt.update({
            'input_folder': LR_folder,
            'save_folder': save_LR_folder,
            'crop_sz': crop_sz // scale_ratio,
            'step': step // scale_ratio,
            'thres_sz': thres_sz // scale_ratio
        })
        extract_single(opt)
        
        assert len(_get_paths_from_images(save_GT_folder)) == len(_get_paths_from_images(save_LR_folder)), \
            'Different length of save_GT_folder and save_LR_folder.'
    else:
        raise ValueError('Wrong mode.')

def extract_single(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir [{save_folder}] ...')
    else:
        print(f'Folder [{save_folder}] already exists. Exit...')
        sys.exit(1)
    img_list = _get_paths_from_images(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')

def worker(path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Wrong image shape - {n_channels}')

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         img_name.replace('.png', f'_s{index:03d}.png')),
                crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']]
            )
    return f'Processing {img_name} ...'

class ProgressBar(object):
    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('Terminal width is too small ({}), please consider widening the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')   # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    assert os.path.isdir(path), f'{path} is not a valid directory'
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, f'{path} has no valid image file'
    return images

if __name__ == '__main__':
    main()
