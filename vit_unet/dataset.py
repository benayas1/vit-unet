import torch
import pydicom
import nibabel as nib

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
        nii_img  = nib.load(path_mask)
        nii_data = nii_img.get_fdata()
        mask = nii_data[:,:,mask_index]
        
        # Augmentation including scaling
        if self.augments:
            augmented = self.augments(image=x, mask=mask)
            x = augmented['image']
            mask = augmented['mask']
            
        return x, mask
        
    def __len__(self):
        return len(self.df)