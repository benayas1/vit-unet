import fire
import vit_unet.torch.model as models
import vit_unet.torch.functions as fn
import vit_unet.torch.dataset as dataset
from glob import glob
import pandas as pd
import torch
from sklearn.model_selection import KFold
import albumentations
import os
import numpy as np
import cv2
import wandb


def main(input_folder='ssid',
         n_epochs=5,
         folds=5,
         model_string='lite',
         lr=0.0001,
         batch_size=8,
         im_size=224):

    WB_ENTITY = 'UAL'
    wandb.login(key='ab1f4c380e0a008223b6434a42907bacfd7b4e26') # WANDB KEY
    with wandb.init(project='ViT_UNet', entity=WB_ENTITY) as run:

        wandb.config.update({'n_epochs':n_epochs,
                             'fold':folds,
                             'model':model_string,
                             'lr':lr
                            })
    
        print('')
        # prepare Data
        clean = np.array([path.split('/')[-1][:-4] for path in sorted(glob('ssid/clean/*'))])
        noisy = np.array([path.split('/')[-1][:-4] for path in sorted(glob('ssid/noisy/*'))])
        
        clean = np.array([path for path in clean if path in noisy])

        assert len(clean)==len(noisy), f"Clean length {len(clean)} is not equal to Noisy length {len(noisy)}"

        cv = KFold(folds, shuffle=True)
        results = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(noisy)):
            train = noisy[train_idx]
            test = noisy[test_idx]

            print(f'FOLD {fold}: Training on {len(train)} samples and testing on {len(test)} samples')
            # Create dataset and dataloader
            train_transform = albumentations.Compose([
                albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                albumentations.Normalize(mean=(0.456), std=(0.224), max_pixel_value=255.0, p=1.0)
            ])

            val_transform = albumentations.Compose([
                albumentations.Normalize(mean=(0.456), std=(0.224), max_pixel_value=255.0, p=1.0)
            ])
            train_dataloader = torch.utils.data.DataLoader(dataset.DenoisingDataset(train, 
                                                                                    clean_folder=os.path.join(input_folder,'clean'), 
                                                                                    noisy_folder=os.path.join(input_folder,'noisy'), 
                                                                                    augments=train_transform,
                                                                                    im_size=im_size),
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=2)
            test_dataloader = torch.utils.data.DataLoader(dataset.DenoisingDataset(test, 
                                                                                   clean_folder=os.path.join(input_folder,'clean'), 
                                                                                   noisy_folder=os.path.join(input_folder,'noisy'), 
                                                                                   augments=val_transform,
                                                                                   im_size=im_size),
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            num_workers=2)

            # Create model
            model = models.get_vit_unet(model_string)
            model.to('cuda')
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # Create fitter
            fitter = dataset.ImageFitter(model,
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
            model.eval()
            score = fn.psnr(model, test_dataloader)
            print(f"FOLD {fold}: Mean PSNR {np.mean(score)}")
            results.append(score)

        print(f"Average Mean PSNR{np.mean(results)}. STD Mean PSNR {np.std(results)}")

        run.log({'psnr_mean':np.mean(results),
                 'psnr_std':np.std(results)})

        run.finish()

if __name__ == '__main__':
    fire.Fire(main)