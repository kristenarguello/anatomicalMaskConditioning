import kornia
import torch
import torch.nn.functional as F


def cnr(pred, target):
    # user-provided definition
    return (pred - target).abs().mean() / (pred + target).abs().mean()


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(
        torch.as_tensor(max_val, device=pred.device) / torch.sqrt(mse)
    )


def ssim_metric(pred, target, window_size=11):
    # kornia returns an SSIM map (B,1,H,W) or scalar; take mean over batch & pixels
    ssim_map = kornia.metrics.ssim(img1=pred, img2=target, window_size=window_size)
    return ssim_map.mean()


def rmse(pred, target):
    # target = gt
    # pred = ld
    return torch.sqrt(F.mse_loss(pred, target))
