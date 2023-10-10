import os
import cv2 as cv 
import torch
from torch.utils.data import Dataset
import numpy as np

img_height = 224
img_width = 224

class data(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array(cv.imread(img_path, cv.IMREAD_GRAYSCALE))
        # image = np.array(cv.imread(img_path))
        mask = np.array(cv.imread(mask_path, cv.IMREAD_GRAYSCALE), dtype=np.float32)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class data_for_ben_or_mal(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(cv.imread(img_path, cv.IMREAD_GRAYSCALE))
        if "benign" in self.images[index]:
            mask = torch.tensor([1,0,0])
        elif "malignant" in self.images[index]:
            mask = torch.tensor([0,1,0])
        elif "normal" in self.images[index]:
            mask = torch.tensor([0,0,1])
        else:
            print("data error")

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        
        return image, mask


