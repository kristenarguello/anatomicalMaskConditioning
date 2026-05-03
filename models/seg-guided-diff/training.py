from dataclasses import dataclass
import math
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

from datetime import timedelta

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import diffusers

from eval import (
    evaluate,
    add_segmentations_to_noise,
    SegGuidedDDPMPipeline,
    LDConditionedDDPMPipeline,
)
from visualization import plot_learning_curve, save_feature_maps


def get_unwrapped_model(model):
    """Get the underlying model, unwrapping DataParallel if needed."""
    return model.module if hasattr(model, 'module') else model


@dataclass
class TrainingConfig:
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    gradient_accumulation_steps: int = 1
    learning_rate: float = 7.5e-5
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 30
    mixed_precision: str = (
        "bf16"  # `no` for float32, `bf16` for automatic mixed precision
    )
    output_dir: str = None

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = (
        True  # overwrite the old model when re-running the notebook
    )
    seed: int = 0

    # custom options
    segmentation_guided: bool = False
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = None  # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "breast_mri"
    resume_epoch: int = None

    # EXPERIMENTAL/UNTESTED: classifier-free class guidance and image translation
    class_conditional: bool = False
    cfg_p_uncond: float = 0.2  # p_uncond in classifier-free guidance paper
    cfg_weight: float = 0.3  # w in the paper

    trans_noise_level: float = (
        0.2  # ratio of time step t to noise trans_start_images to total T before denoising in translation. e.g. value of 0.2 means t = 200 for default T = 1000.
    )
    use_cfg_for_eval_conditioning: bool = (
        True  # whether to use classifier-free guidance for or just naive class conditioning for main sampling loop
    )
    cfg_maskguidance_condmodel_only: bool = (
        True  # if using mask guidance AND cfg, only give mask to conditional network
    )
    # ^ this is because giving mask to both uncond and cond model make class guidance not work
    # (see "Classifier-free guidance resolution weighting." in ControlNet paper)


def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    eval_dataloader,
    lr_scheduler,
    device="cuda",
    start_global_step=0,
):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    global_step = start_global_step

    # logging (TensorBoard under output_dir for easy comparison across runs)
    run_name = "{}-{}-{}".format(
        config.model_type.lower(), config.dataset, config.image_size
    )
    if config.segmentation_guided:
        run_name += "-segguided"
    log_dir = os.path.join(config.output_dir, "tensorboard", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Iterator over eval set; reset when exhausted so long runs can keep evaluating
    eval_dataloader_iter = iter(eval_dataloader)

    # history for learning curve and CSV log at end of training
    history = {"step": [], "loss": [], "lr": []}

    # Now you train the model
    start_epoch = 0
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch

    # AMP setup
    if config.mixed_precision == "bf16":
        use_amp = True
        amp_dtype = torch.bfloat16
        use_scaler = False  # bf16 typically does not need gradient scaling
    elif config.mixed_precision == "fp16":
        use_amp = True
        amp_dtype = torch.float16
        use_scaler = True
    else:
        use_amp = False
        amp_dtype = torch.float32
        use_scaler = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # type: ignore

    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        print(f"Epoch {epoch}")

        model.train()

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device, non_blocking=True)
            lds_images = batch["lds_cond"].to(device, non_blocking=True)

            # Sample noise to add to the images
            noise = torch.randn_like(clean_images, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config.segmentation_guided:
                noisy_images = add_segmentations_to_noise(
                    noisy_images, batch, config, device
                )
            # ===== forward + loss (main branch) =====
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if config.class_conditional:
                    class_labels = torch.ones(
                        noisy_images.size(0), device=device
                    ).long()
                    # classifier-free guidance drop
                    if np.random.uniform() <= config.cfg_p_uncond:
                        class_labels = torch.zeros_like(class_labels)
                    noise_pred = model(
                        noisy_images,
                        timesteps,
                        class_labels=class_labels,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                loss_main = F.mse_loss(noise_pred, noise)

                # ===== optional target domain branch (same accumulation window) =====
                loss_target_domain = None
                if config.class_conditional:
                    target_domain_images = batch["images_target"].to(
                        device, non_blocking=True
                    )
                    noise_td = torch.randn_like(
                        target_domain_images, device=target_domain_images.device
                    )
                    bs_td = target_domain_images.size(0)
                    timesteps_td = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bs_td,),
                        device=target_domain_images.device,
                    ).long()
                    noisy_td = noise_scheduler.add_noise(
                        target_domain_images, noise_td, timesteps_td
                    )

                    if config.segmentation_guided:
                        # no masks in target domain → blank mask
                        noisy_td = torch.cat(
                            (noisy_td, torch.zeros_like(noisy_td)), dim=1
                        )

                    class_labels_td = torch.full(
                        [noisy_td.size(0)], 2, device=device
                    ).long()
                    if np.random.uniform() <= config.cfg_p_uncond:
                        class_labels_td = torch.zeros_like(class_labels_td)
                    noise_pred_td = model(
                        noisy_td,
                        timesteps_td,
                        class_labels=class_labels_td,
                        return_dict=False,
                    )[0]
                    loss_target_domain = F.mse_loss(noise_pred_td, noise_td)

                # total loss for this micro-step
                total_loss = (
                    loss_main
                    if loss_target_domain is None
                    else (loss_main + loss_target_domain)
                )

                # scale down for accumulation
                total_loss = total_loss / config.gradient_accumulation_steps

            # backward (accumulation)
            scaler.scale(total_loss).backward()

            # step only every accumulation window or at the very end
            do_update = ((step + 1) % config.gradient_accumulation_steps == 0) or (
                (step + 1) == len(train_dataloader)
            )
            if do_update:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if hasattr(optimizer, "step_with_scaler"):
                    # HybridMuonAdamW: Muon needs manual unscaling; AdamW goes via scaler
                    optimizer.step_with_scaler(scaler)
                else:
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1
                history["step"].append(global_step)
                history["loss"].append(loss_main.detach().item())
                history["lr"].append(lr_scheduler.get_last_lr()[0])

            # logging/progress
            if config.class_conditional and (loss_target_domain is not None):
                logs = {
                    "loss": loss_main.detach().item(),
                    "loss_target_domain": loss_target_domain.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "upd_step": global_step,
                }
                writer.add_scalar(
                    "loss_target_domain", loss_target_domain.detach().item(), global_step
                )
            else:
                logs = {
                    "loss": loss_main.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "upd_step": global_step,
                }
            writer.add_scalar("loss", loss_main.detach().item(), global_step)
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        unwrapped_model = get_unwrapped_model(model)
        if config.model_type == "DDPM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDPMPipeline(
                    unet=unwrapped_model,
                    scheduler=noise_scheduler,
                    eval_dataloader=eval_dataloader,
                    external_config=config,
                )
            else:
                if config.class_conditional:
                    raise NotImplementedError(
                        "TODO: Conditional training not implemented for non-seg-guided DDPM"
                    )
                else:
                    pipeline = LDConditionedDDPMPipeline(
                        unet=unwrapped_model,
                        scheduler=noise_scheduler,
                        external_config=config,
                    )

        model.eval()

        if (
            epoch + 1
        ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            try:
                seg_batch = next(eval_dataloader_iter)
            except StopIteration:
                eval_dataloader_iter = iter(eval_dataloader)
                seg_batch = next(eval_dataloader_iter)
            if config.segmentation_guided:
                evaluate(
                    config,
                    epoch,
                    pipeline,
                    seg_batch=seg_batch,
                    class_label_cfg=None,
                    translate=True,
                    device=device,
                    writer=writer,
                    num_inference_steps=1000,  # tweak as desired
                )

            else:
                evaluate(config, epoch, pipeline, seg_batch=seg_batch)

        if (
            epoch + 1
        ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline.save_pretrained(config.output_dir)
            # Save full training checkpoint (optimizer + scheduler state)
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(config.output_dir, "checkpoint.pt"))
            print(f"Saved checkpoint at epoch {epoch + 1}, global_step {global_step}")

    # End of training: save learning curve and optional feature maps
    plot_learning_curve(history, config.output_dir, smooth=min(50, max(1, len(history["step"]) // 20)))
    try:
        viz_batch = next(iter(train_dataloader))
        save_feature_maps(
            unwrapped_model,
            viz_batch,
            device,
            os.path.join(config.output_dir, "feature_maps"),
            config,
            noise_scheduler,
            add_segmentations_to_noise,
            max_channels=32,
        )
    except Exception as e:
        print(f"Could not save feature maps: {e}")

    writer.close()
