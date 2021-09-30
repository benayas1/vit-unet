import torch
import pydicom
import nibabel as nib
import cv2
import os
import numpy as np
from benatools.torch.fitter import TorchFitterBase

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, df, augments=None, is_test=False, data_folder='output', im_size=(128,128), ls=0.0):
        self.df = df
        self.augments = augments
        self.is_test = is_test
        self.data_folder = data_folder
        self.im_size = im_size
        self.ls = ls
        
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        path_image = row['image']
        path_mask = row['mask']
        mask_index = row['mask_index']

        # Image
        x = pydicom.read_file(path_image).pixel_array

        # Mask
        nii_img = nib.load(path_mask)
        nii_data = nii_img.get_fdata()
        mask = nii_data[:,:,mask_index]
        
        # Augmentation including scaling
        if self.augments:
            augmented = self.augments(image=x, mask=mask)
            x = augmented['image']
            mask = augmented['mask']
            
        return {'x':x, 'y':mask}
        
    def __len__(self):
        return len(self.df)


class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, img_names, augments=None, clean_folder='/ssid/clean/', noisy_folder='/ssid/noisy/'):
        self.img_names = img_names
        self.augments = augments
        self.clean_folder = clean_folder
        self.noisy_folder = noisy_folder
        
    def __getitem__(self, idx):
        img_clean = cv2.imread(os.path.join(self.clean_folder,self.img_names[idx])+'.png')
        img_noisy = cv2.imread(os.path.join(self.noisy_folder,self.img_names[idx])+'.png')
        
        # Augmentation including scaling
        if self.augments:
            augmented = self.augments(image=img_clean, mask=img_noisy)
            img_clean = augmented['image']
            img_noisy = augmented['mask']

        img_noisy = img_noisy.transpose(2,0,1).astype(np.float)
        img_clean = img_clean.transpose(2,0,1).astype(np.float)
            
        return {'x':img_noisy, 'y':img_clean}
        
    def __len__(self):
        return len(self.img_names)


class ImageFitter(TorchFitterBase):

    def unpack(self, data):
        # extract x and y from the dataloader
        x = data['x'].to(self.device).float()
        y = data['y'].to(self.device).float()

        # weights if existing
        if 'w' in data:
            w = data['w']
            w = w.to(self.device)
            w = w.float()
        else:
            w = None

        return x, y, w