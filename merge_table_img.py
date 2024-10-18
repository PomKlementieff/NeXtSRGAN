import cv2
import numpy as np

for name in ['baby', 'bird', 'butterfly', 'head', 'woman']:
    img1 = cv2.imread(f'./results/psnr_pretrain/set5/Bic_SR_HR_{name}.png')
    img2 = cv2.imread(f'./results/nextsrgan/set5/Bic_SR_HR_{name}.png')
    
    padd = np.zeros([img1.shape[0], 5, 3], np.uint8)
    img_merge = np.concatenate((padd, img1, padd, img2, padd), 1)
    
    padd = np.zeros([5, img_merge.shape[1], 3], np.uint8)
    img_merge = np.concatenate((padd, img_merge, padd), 0)
    
    cv2.imwrite(f'./photo/table_{name}.png', img_merge)