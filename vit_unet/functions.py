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


def softmax_top(X:torch.Tensor,
                top:int,
                dim:int = -1,
                ):
    X = X.clone()
    #print('Original tensor:')
    #print(X)
    #print('Original softmax:')
    #print(torch.nn.functional.softmax(X, dim = -1))
    values, idx = torch.topk(X, top, dim = dim)
    values = torch.nn.functional.softmax(values, dim = dim)
    axis = torch.unsqueeze(torch.as_tensor([row for row in range(X.shape[-2]) for _ in range(top)]),0)
    idx = torch.unsqueeze(idx.flatten(), 0)
    idx = torch.cat([axis,idx], dim = 0)
    Y = torch.sparse.FloatTensor(idx, values.flatten(), X.size()).to_dense()
    #print('Top-{} softmax:'.format(str(top)))
    #print(Y)
    return Y