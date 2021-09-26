# ViT-UNet
ViT-UNet is a novel hierarchical ViT-based model, applied to autoencoders via UNet-shaped architectures. Background work can be found in the folowing links:
* [Deep-ViT](https://arxiv.org/pdf/2103.11886.pdf)
* [UNet](https://arxiv.org/pdf/1505.04597.pdf)

This Autoencoder structure aims to take advantage of the computational parallelisation of self-attention mechanisms, at the same time that 

![Model architecture illustration](https://github.com/benayas1/vit-unet/blob/main/sample_images/architecture.PNG)

The following tasks are to be tested:
1. Image denoising.
    * Dataset: [SIDD dataset](https://paperswithcode.com/sota/image-denoising-on-sidd).
    * Two models are outstanding in the classification, which are [HINet](https://paperswithcode.com/paper/hinet-half-instance-normalization-network-for) (best model in PSNR metric) and [UFormer](https://paperswithcode.com/paper/uformer-a-general-u-shaped-transformer-for) (best model in SSIM metric).
2. Deblurring.
    * Dataset: [GoPro dataset](https://paperswithcode.com/dataset/gopro).
    * The top model is [HINet](https://paperswithcode.com/paper/hinet-half-instance-normalization-network-for) with PSNR metric.
3. Single Image Deraining.
    * Multiple datasets available: Rain110H, Rain110L,... the full list can be found [here](https://paperswithcode.com/dataset/synthetic-rain-datasets).

4. Image segmentation.
    * Dataset: [Pancreas Segmentation on TCIA Pancreas-CT](https://paperswithcode.com/sota/pancreas-segmentation-on-tcia-pancreas-ct). The metric that is used here is the Dice Score, which is the equivalent to F1 w.r.t. accuracy in image segmentation, corresponding the latter to Jaccard index (IoU). A softer version of this index can be explored [here](https://arxiv.org/abs/1911.01685).

Metrics that are required for these tasks:
1. [Peak signal-to-noise ratio (PSNR)](https://pytorch.org/ignite/generated/ignite.metrics.PSNR.html)
2. [Strictural Similarity (SSIM)](https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html)
3. Soft Dice Score: 
```
def dice_loss(input:torch.Tensor,
              target:torch.Tensor,
              ):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
```