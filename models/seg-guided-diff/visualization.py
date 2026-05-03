"""
Training visualizations: learning curves, feature maps, and CSV logs for later comparison.
"""

import csv
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def plot_learning_curve(
    history: Dict[str, List[float]],
    output_dir: str,
    smooth: Optional[int] = None,
) -> None:
    """
    Plot loss (and optionally LR) vs step and save to output_dir.
    Also saves training_log.csv for later comparison with other runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    steps = history["step"]
    if not steps:
        return

    # Save CSV for external comparison
    csv_path = os.path.join(output_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "lr"])
        for i in range(len(steps)):
            w.writerow([
                steps[i],
                history["loss"][i],
                history["lr"][i],
            ])

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    loss = np.array(history["loss"])
    if smooth and len(loss) >= smooth:
        kernel = np.ones(smooth) / smooth
        loss_smooth = np.convolve(loss, kernel, mode="valid")
        steps_smooth = steps[smooth - 1:]
        ax1.plot(steps_smooth, loss_smooth, "b-", linewidth=1.5, label=f"loss (smoothed {smooth})")
    else:
        ax1.plot(steps, loss, "b-", alpha=0.6, linewidth=0.8, label="loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, history["lr"], "g-", alpha=0.5, linewidth=0.6, label="lr")
    ax2.set_ylabel("Learning rate", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85))

    plt.title("Learning curve")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curve and {csv_path}")


def _make_feature_grid(feats: torch.Tensor, max_channels: int = 64, nrow: int = 8) -> np.ndarray:
    """Turn a feature tensor (B, C, H, W) into a single grid image for saving."""
    feats = feats.detach().float().cpu()
    b, c, h, w = feats.shape
    c = min(c, max_channels)
    feats = feats[:1, :c]  # first sample, first c channels
    # normalize per channel for visibility
    for i in range(c):
        ch = feats[0, i]
        mn, mx = ch.min().item(), ch.max().item()
        if mx - mn > 1e-6:
            feats[0, i] = (ch - mn) / (mx - mn)
    # (1, c, h, w) -> (c, h, w), then grid
    feats = feats[0]
    ncol = (c + nrow - 1) // nrow
    rows = []
    for i in range(ncol):
        row = feats[i * nrow : (i + 1) * nrow]
        if row.shape[0] < nrow:
            row = torch.cat([row, torch.zeros(nrow - row.shape[0], h, w)], dim=0)
        rows.append(row)
    grid = torch.cat(rows, dim=1)
    grid = grid.permute(1, 2, 0).numpy()
    if grid.shape[2] == 1:
        grid = grid.squeeze(-1)
    return np.clip(grid, 0, 1)


def save_feature_maps(
    unwrapped_model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    config: Any,
    noise_scheduler: Any,
    add_segmentations_to_noise_fn,
    max_channels: int = 32,
) -> None:
    """
    Run one forward pass, collect activations from selected layers, and save as images.
    Useful to compare feature responses across different training runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    unwrapped_model.eval()

    # Build input like in training (one sample to save memory)
    clean_images = batch["images"].to(device, non_blocking=True)[:1]
    lds_images = batch["lds_cond"].to(device, non_blocking=True)[:1]
    noise = torch.randn_like(clean_images, device=device)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (1,),
        device=device,
    ).long()
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    noisy_images = torch.cat((noisy_images, lds_images), dim=1)
    if config.segmentation_guided:
        batch_small = {k: v[:1].to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        noisy_images = add_segmentations_to_noise_fn(noisy_images, batch_small, config, device)

    activations: Dict[str, torch.Tensor] = {}

    def get_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                activations[name] = out.detach()
            else:
                activations[name] = out[0].detach() if isinstance(out[0], torch.Tensor) else None
        return hook

    # Register hooks on a few representative layers (UNet2D naming from diffusers)
    hooks = []
    try:
        if hasattr(unwrapped_model, "conv_in"):
            hooks.append(unwrapped_model.conv_in.register_forward_hook(get_hook("conv_in")))
        if hasattr(unwrapped_model, "down_blocks") and len(unwrapped_model.down_blocks) > 0:
            d0 = unwrapped_model.down_blocks[0]
            if hasattr(d0, "resnets") and len(d0.resnets) > 0:
                hooks.append(d0.resnets[0].register_forward_hook(get_hook("down_block_0_resnet_0")))
        if hasattr(unwrapped_model, "down_blocks") and len(unwrapped_model.down_blocks) > 2:
            d2 = unwrapped_model.down_blocks[2]
            if hasattr(d2, "resnets") and len(d2.resnets) > 0:
                hooks.append(d2.resnets[0].register_forward_hook(get_hook("down_block_2_resnet_0")))
    except Exception as e:
        print(f"Could not register some feature map hooks: {e}")

    try:
        with torch.no_grad():
            _ = unwrapped_model(noisy_images, timesteps, return_dict=False)
    finally:
        for h in hooks:
            h.remove()

    for name, feats in activations.items():
        if feats is None:
            continue
        try:
            grid = _make_feature_grid(feats, max_channels=max_channels)
            path = os.path.join(output_dir, f"feature_maps_{name}.png")
            if grid.ndim == 2:
                plt.imsave(path, grid, cmap="viridis")
            else:
                plt.imsave(path, grid)
            print(f"Saved {path}")
        except Exception as e:
            print(f"Could not save feature map {name}: {e}")
