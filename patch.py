import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import cv2
import os

# Read images and labels directories
images_path = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/images/train'
masks_path = '/mnt/c/projects/unet-multiclass/krovovi/Standarised/labels/train'

counter = 0
# Patchify images 
for filename in os.listdir(images_path):
   img_path = os.path.join(images_path, filename)
   img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
   
   # (256, 256, 3)
   patches_img = patchify(img, (256, 256, 3), step=256)

   for i in range(patches_img.shape[0]):
       for j in range(patches_img.shape[1]):
           # [i, j, 0, :, :]
           single_patch_img = patches_img[i, j, 0, :, :, :]
           
           counter += 1
           
           patch_filename = f'image_{counter}_{i}_{j}.png'
           patch_save_path = os.path.join('/mnt/c/projects/unet-multiclass/krovovi/Standarised/patches2/images', patch_filename)
           cv2.imwrite(patch_save_path, single_patch_img)
           
print('Shape of single patch ', single_patch_img.shape)
counter = 0

# Patchify labels 
for filename in os.listdir(masks_path):
    msk_path = os.path.join(masks_path, filename)
    msk = cv2.imread(msk_path,cv2.IMREAD_GRAYSCALE)
    
    patches_msk = patchify(msk, (256, 256), step=256)
    
    for i in range(patches_msk.shape[0]):
        for j in range(patches_msk.shape[1]):
            single_patch_msk = patches_msk[i, j, :, :]
                    
            counter += 1
                    
            patch_filename = f'mask_{counter}_{i}_{j}.png'
            patch_save_path = os.path.join('/mnt/c/projects/unet-multiclass/krovovi/Standarised/patches2/labels', patch_filename)
            cv2.imwrite(patch_save_path, single_patch_msk)
print('Shape of single patch ', single_patch_msk.shape)
