import torch
from skimage.metrics import peak_signal_noise_ratio
import numpy as np


def psnr(model, dataloader):
    # Calculate PSNR
    score = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to('cuda')
            output = model(**x).cpu().numpy()
            y = batch['y'].numpy()

            for i in range(len(output)):
                score.append(peak_signal_noise_ratio(y[i], output[i]))
    score = np.array(score)
    return score
