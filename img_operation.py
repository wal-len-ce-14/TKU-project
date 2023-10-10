import os
import cv2 as cv
import numpy as np
import shutil

def merge_mask(mask_folder):
    # 將一樣圖片的mask合併成一個mask
    masks = os.listdir(mask_folder)
    for i, file in enumerate(masks):
        if 'mask_' in file:
            img_copy = cv.imread(os.path.join(mask_folder, file))
            f = file.split('_')
            img = cv.imread(os.path.join(mask_folder, f[0]+"_"+f[1]+".png"))
            print(os.path.join(mask_folder, f[0]+"_"+f[1]+".png"),"\n")   
            img = np.clip(img + img_copy, 0, 255)
            cv.imwrite(os.path.join(mask_folder, f[0]+"_"+f[1]+".png") ,img)
            os.remove(os.path.join(mask_folder, file))

def classification(source_dir, original_dir, masks_dir):
    # 將有原本圖片以及mask圖片的資料夾分成 img資料夾 跟 mask資料夾 
    all_file = os.listdir(source_dir)
    o = 1
    m = 1
    for i, file in enumerate(all_file):
        if 'mask' in file:
            shutil.copy(os.path.join(source_dir, file), masks_dir)
            m += 1

            print(f"mask in {file}")
        else:
            shutil.copy(os.path.join(source_dir, file), original_dir)
            o += 1

            print(f"not in {file}")