import fire
from vit_unet import models, dataset
import vit_unet.functions as fn
from glob import glob
import pandas as pd
import torch
import sklearn
from benatools.torch.fitter import ImageFitter
import albumentations
import os
import numpy as np
import cv2
import wandb


def main(input_folder='ssid',
         n_epochs=5,
         folds=5,
         model_string='lite',
         lr=0.0001):

    WB_ENTITY = 'UAL'
    wandb.login(key='ab1f4c380e0a008223b6434a42907bacfd7b4e26') # WANDB KEY
    with wandb.init() as run:

        wandb.config.update({'n_epochs':n_epochs,
                             'fold':folds,
                             'model':model_string,
                             'lr':lr
                            })
    
        # prepare Data
        clean = np.array([path.split('/')[:-4] for path in sorted(glob('ssid/clean/*'))])
        noisy = np.array([path.split('/')[:-4] for path in sorted(glob('ssid/noisy/*'))])
        
        clean = np.array([path for path in clean if path in noisy])

        assert len(clean)==len(noisy)

        cv = sklearn.model_selection.KFold(folds, shuffle=True)
        results = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(noisy)):
            train = noisy[train_idx]
            test = noisy[test_idx]

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
            model = models.get_vit_unet(model_string)
            model.to('cuda')
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # Create fitter
            fitter = ImageFitter(model,
                                 loss=criterion,
                                 optimizer=optimizer,
                                 device='cuda',
                                 folder='models')

            def wandb_update(x):
                data_log = x.copy()
                del data_log['epoch']
                wandb.log({'training_'+str(fold):data_log})

            history = fitter.fit(train_dataloader,
                                 test_dataloader,
                                 n_epochs=n_epochs,
                                 callbacks=[wandb_update])

            fitter.load('models/best-checkpoint.bin')

            # Calculate PSNR
            model = fitter.model
            score = fn.psnr(model, test_dataloader)
            print(f"FOLD {fold}: Mean PSNR {np.mean(score)}")
            results.append(score)

        print(f"Average Mean PSNR{np.mean(results)}. STD Mean PSNR {np.std(results)}")

        run.log({'psnr_mean':np.mean(results),
                 'psnr_std':np.std(results)})

        run.finish()

if __name__ == '__main__':
    fire.Fire(main)