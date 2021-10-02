import torch
from skimage.metrics import peak_signal_noise_ratio
import itertools
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


def softmax_top(X:torch.Tensor,
                top:int,
                ):
    batch_size, channels, shape, _ = X.size()
    X = X.clone()
    values, idx = torch.topk(X, top, dim = -1)
    values = torch.nn.functional.softmax(values, dim = -1)
    idx = torch.unsqueeze(idx.flatten(), 0)
    dct_axis = torch.as_tensor(list(itertools.product(range(batch_size),range(channels),range(shape))), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    axis = torch.stack([elem.T for elem in dct_axis for _ in range(top)], dim = -1)
    idx = torch.cat([axis,idx], dim = 0)
    Y = torch.sparse.FloatTensor(idx, values.flatten(), X.size()).to_dense()
    return Y