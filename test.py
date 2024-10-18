from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf

from modules.nextsrgan import RRDB_Model
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)

flags.DEFINE_string('cfg_path', './configs/nextsrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')

def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = RRDB_Model(None, cfg['ch_size'], cfg['network_G'])

    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print(f"[*] load ckpt from {tf.train.latest_checkpoint(checkpoint_dir)}.")
    else:
        print(f"[*] Cannot find ckpt from {checkpoint_dir}")
        exit()

    if FLAGS.img_path:
        print(f"[*] Processing on single image {FLAGS.img_path}")
        raw_img = cv2.imread(FLAGS.img_path)
        lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale'])

        sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
        bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)

        str_format = "[{}] PSNR/SSIM: Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
        print(str_format.format(
            os.path.basename(FLAGS.img_path),
            calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
            calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
            calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
            calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
        result_img_path = './Bic_SR_HR_' + os.path.basename(FLAGS.img_path)
        print(f"[*] write the result image {result_img_path}")
        results_img = np.concatenate((bic_img, sr_img, hr_img), 1)
        cv2.imwrite(result_img_path, results_img)
    else:
        print("[*] Processing on Set5 and Set14, and write results")
        results_path = './results/' + cfg['sub_name'] + '/'
        set5_list = []
        set14_list = []

        for key, path in cfg['test_dataset'].items():
            print(f"'{key}' from {path}\n  PSNR/SSIM")
            dataset_name = key.replace('_path', '')
            pathlib.Path(results_path + dataset_name).mkdir(parents=True, exist_ok=True)
            
            for img_name in os.listdir(path):
                raw_img = cv2.imread(os.path.join(path, img_name))
                lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale'])

                sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
                bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)

                str_format = "  [{}] Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
                result_str = str_format.format(
                    img_name + ' ' * max(0, 20 - len(img_name)),
                    calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                    calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                    calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
                    calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)))
                print(result_str)

                if dataset_name == 'set5':
                    set5_list.append(result_str)
                else:
                    set14_list.append(result_str)

                result_img_path = os.path.join(results_path + dataset_name, 'Bic_SR_HR_' + img_name)
                results_img = np.concatenate((bic_img, sr_img, hr_img), 1)
                cv2.imwrite(result_img_path, results_img)
        print(f"[*] write the visual results in {results_path}")

        with open(f'./results/{cfg["sub_name"]}/test_results({cfg["sub_name"]}).txt', 'w') as f:
            f.write('##### Test Results #####\n\n')
            f.write(f'·Set5: \n{str(set5_list)[1:-1]}\n\n')
            f.write(f'·Set14: \n{str(set14_list)[1:-1]}\n')

        set5_psnr = []
        set5_ssim = []
        set14_psnr = []
        set14_ssim = []
        
        for result in set5_list:
            set5_psnr.append(float(result.split('SR=')[1].split('db')[0]))
            set5_ssim.append(float(result.split('/')[-1]))

        for result in set14_list:
            set14_psnr.append(float(result.split('SR=')[1].split('db')[0]))
            set14_ssim.append(float(result.split('/')[-1]))

        set5_df = pd.DataFrame(zip(set5_psnr, set5_ssim), columns=['Set5_PSNR', 'Set5_SSIM'])
        set5_df.to_excel(f'./results/{cfg["sub_name"]}/test_results({cfg["sub_name"]}_set5).xlsx', index=False)

        set14_df = pd.DataFrame(zip(set14_psnr, set14_ssim), columns=['Set14_PSNR', 'Set14_SSIM'])
        set14_df.to_excel(f'./results/{cfg["sub_name"]}/test_results({cfg["sub_name"]}_set14).xlsx', index=False)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass