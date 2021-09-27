import fire
from vit_unet import models, dataset
from glob import glob
import pandas as pd
import torch
import sklearn
from benatools.torch.fitter import ImageFitter
import albumentations
import os
import cv2

def run(input_folder='ssid',
        n_epochs=5):
    
    # prepare Data
    clean = [path.split('/')[:-4] for path in sorted(glob('ssid/clean/*'))]
    noisy = [path.split('/')[:-4] for path in sorted(glob('ssid/noisy/*'))]
    
    clean = [path for path in clean if path in noisy]

    assert len(clean)==len(noisy)

    # train/test split
    train, test = sklearn.model_selection.train_test_split(test_size=0.1)

    # Create dataset and dataloader
    train_transform = albumentations.Compose([
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        albumentations.Normalize(mean=(0.456), std=(0.224), max_pixel_value=255.0, p=1.0)
    ])

    val_transform = albumentations.Compose([
        albumentations.Normalize(mean=(0.456), std=(0.224), max_pixel_value=255.0, p=1.0)
    ])
    batch_size = 8
    train_dataloader = torch.utils.data.Dataloader(dataset.DenoisingDataset(train, 
                                                                            clean_folder=os.path.join(input_folder,'clean'), 
                                                                            noisy_folder=os.path.join(input_folder,'noisy'), 
                                                                            augments=train_transform),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
    test_dataloader = torch.utils.data.Dataloader(dataset.DenoisingDataset(test, 
                                                                           clean_folder=os.path.join(input_folder,'clean'), 
                                                                           noisy_folder=os.path.join(input_folder,'noisy'), 
                                                                           augments=val_transform),
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=2)

    # Create model
    model = models.ViT_UNet()
    model.to('cuda')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # Create fitter
    fitter = ImageFitter(model,
                         loss=criterion,
                         optimizer=optimizer,
                         device='cuda')

    fitter.fit(train_dataloader,
               test_dataloader,
               n_epochs=n_epochs)


if __name__ == '__main__':
    run()