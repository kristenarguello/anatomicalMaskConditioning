import atexit
import math
import os
import sys
from argparse import ArgumentParser

# torch imports
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# HF imports
import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import datasets

# custom imports
from training import TrainingConfig, train_loop
from eval import evaluate_generation, evaluate_sample_many
from utils import get_all_images_paths
from muon import HybridMuonAdamW, get_muon_and_adamw_params


class Tee:
    """Write to both a stream (e.g. stdout) and a file."""

    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def writable(self):
        return True


def main(
    mode,
    img_size,
    num_img_channels,
    dataset,
    img_dir,
    seg_dir,
    model_type,
    segmentation_guided,
    segmentation_channel_mode,
    num_segmentation_classes,
    train_batch_size,
    eval_batch_size,
    num_epochs,
    learning_rate,
    resume_epoch=None,
    use_ablated_segmentations=False,
    eval_shuffle_dataloader=True,
    # arguments only used in eval
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000,
    gradient_accumulation_steps=1,
    pretrained_model_path=None,
    comparison=False,
    argv=None,
):
    # load config
    output_dir = "{}-{}-{}".format(
        model_type.lower(), dataset, img_size
    )  # the model namy locally and on the HF Hub
    if segmentation_guided:
        output_dir += "-segguided"
        assert (
            seg_dir is not None
        ), "must provide segmentation directory for segmentation guided training/sampling"

    if use_ablated_segmentations or eval_mask_removal or eval_blank_mask:
        output_dir += "-ablated"

    # Create output dir and set up run logging (command + terminal output)
    os.makedirs(output_dir, exist_ok=True)
    if argv is not None:
        run_command_path = os.path.join(output_dir, "run_command.txt")
        with open(run_command_path, "w") as f:
            f.write(" ".join(argv) + "\n")
    run_log_path = os.path.join(output_dir, "run_log.txt")
    _run_log_file = open(run_log_path, "w", encoding="utf-8")
    atexit.register(_run_log_file.close)
    sys.stdout = Tee(sys.stdout, _run_log_file)
    sys.stderr = Tee(sys.stderr, _run_log_file)

    # GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on {}".format(device))

    print("output dir: {}".format(output_dir))

    if mode == "train":
        evalset_name = "val"
        assert img_dir is not None, "must provide image directory for training"
    elif "eval" in mode:
        evalset_name = "test"

    print("using evaluation set: {}".format(evalset_name))

    config = TrainingConfig(
        image_size=img_size,
        dataset=dataset,
        segmentation_guided=segmentation_guided,
        segmentation_channel_mode=segmentation_channel_mode,
        num_segmentation_classes=num_segmentation_classes,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_dir=output_dir,
        model_type=model_type,
        resume_epoch=resume_epoch,
        use_ablated_segmentations=use_ablated_segmentations,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    load_images_as_np_arrays = False
    if num_img_channels not in [1, 3]:
        # image pil only accepts 1 or 3 channels (greyscales or RGB)
        # treat as numpy array any other amount of channels
        load_images_as_np_arrays = True
        print("image channels not 1 or 3, attempting to load images as np arrays...")

    if config.segmentation_guided:
        seg_types = os.listdir(
            seg_dir
        )  # will list the inner folders (mask title for each one)
        seg_paths_train = {}
        seg_paths_eval = {}

        # train set
        if img_dir is not None:
            # make sure the images are matched to the segmentation masks
            img_dir_train = os.path.join(img_dir, "train")
            img_paths_train = get_all_images_paths(img_dir_train, dose="FD")
            ld_paths_train = get_all_images_paths(img_dir_train, dose="QD")

            for seg_type in seg_types:
                seg_paths_train[seg_type] = get_all_images_paths(
                    os.path.join(seg_dir, seg_type, "train"),
                    dose="QD",  # masks will always be generated using the QD images
                )
        else:
            raise NotImplementedError("training without images not supported!")

        # eval set
        if img_dir is not None:
            img_dir_eval = os.path.join(img_dir, evalset_name)
            img_paths_eval = get_all_images_paths(img_dir_eval, dose="FD")
            ld_paths_eval = get_all_images_paths(img_dir_eval, dose="QD")

            for seg_type in seg_types:
                seg_paths_eval[seg_type] = get_all_images_paths(
                    os.path.join(seg_dir, seg_type, evalset_name),
                    dose="QD",  # masks will always be generated using the QD images
                )
        else:
            raise NotImplementedError("evaluation without images not supported!")

        if img_dir is not None:
            dset_dict_train = {
                **{"image": img_paths_train},
                **{"lds_cond": ld_paths_train},
                **{
                    "seg_{}".format(seg_type): seg_paths_train[seg_type]
                    for seg_type in seg_types
                },
            }

            dset_dict_eval = {
                **{"image": img_paths_eval},
                **{"lds_cond": ld_paths_eval},
                **{
                    "seg_{}".format(seg_type): seg_paths_eval[seg_type]
                    for seg_type in seg_types
                },
            }
        else:
            raise NotImplementedError(
                "training/evaluation without images not supported!"
            )

        if img_dir is not None:
            # add image filenames to dataset (for the img and the lds)
            dset_dict_train["image_filename"] = [
                os.path.basename(f) for f in dset_dict_train["image"]
            ]
            dset_dict_train["lds_cond_filename"] = [
                os.path.basename(f) for f in dset_dict_train["lds_cond"]
            ]

            dset_dict_eval["image_filename"] = [
                os.path.basename(f) for f in dset_dict_eval["image"]
            ]
            dset_dict_eval["lds_cond_filename"] = [
                os.path.basename(f) for f in dset_dict_eval["lds_cond"]
            ]
        else:
            raise NotImplementedError(
                "training/evaluation without images not supported!"
            )

        dataset_train = datasets.Dataset.from_dict(dset_dict_train)
        dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

        # load the images
        if not load_images_as_np_arrays and img_dir is not None:
            dataset_train = dataset_train.cast_column("image", datasets.Image())
            dataset_train = dataset_train.cast_column("lds_cond", datasets.Image())

            dataset_eval = dataset_eval.cast_column("image", datasets.Image())
            dataset_eval = dataset_eval.cast_column("lds_cond", datasets.Image())

        for seg_type in seg_types:
            dataset_train = dataset_train.cast_column(
                "seg_{}".format(seg_type), datasets.Image()
            )

        for seg_type in seg_types:
            dataset_eval = dataset_eval.cast_column(
                "seg_{}".format(seg_type), datasets.Image()
            )

    else:
        # this will only be used as baseline with no masks, but still on ldct =
        # no segmentation, means only conditioning is ldct
        if img_dir is not None:
            img_dir_train = os.path.join(img_dir, "train")
            img_paths_train = get_all_images_paths(img_dir_train, dose="FD")
            ld_paths_train = get_all_images_paths(img_dir_train, dose="QD")

            img_dir_eval = os.path.join(img_dir, evalset_name)
            img_paths_eval = get_all_images_paths(img_dir_eval, dose="FD")
            ld_paths_eval = get_all_images_paths(img_dir_eval, dose="QD")

            dset_dict_train = {
                **{"image": img_paths_train},
                **{"lds_cond": ld_paths_train},
            }

            dset_dict_eval = {
                **{"image": img_paths_eval},
                **{"lds_cond": ld_paths_eval},
            }

            # add image filenames to dataset
            dset_dict_train["image_filename"] = [
                os.path.basename(f) for f in dset_dict_train["image"]
            ]
            dset_dict_eval["image_filename"] = [
                os.path.basename(f) for f in dset_dict_eval["image"]
            ]

            # add ld image paths
            dset_dict_train["lds_cond_filename"] = [
                os.path.basename(f) for f in dset_dict_train["lds_cond"]
            ]
            dset_dict_eval["lds_cond_filename"] = [
                os.path.basename(f) for f in dset_dict_eval["lds_cond"]
            ]

            dataset_train = datasets.Dataset.from_dict(dset_dict_train)
            dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

            # load the images
            if not load_images_as_np_arrays:
                dataset_train = dataset_train.cast_column("image", datasets.Image())
                dataset_eval = dataset_eval.cast_column("image", datasets.Image())

                # ld conditioning
                dataset_train = dataset_train.cast_column("lds_cond", datasets.Image())
                dataset_eval = dataset_eval.cast_column("lds_cond", datasets.Image())

    # training set preprocessing
    if not load_images_as_np_arrays:
        preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                # transforms.RandomHorizontalFlip(), # flipping wouldn't result in realistic images
                transforms.ToTensor(),
                transforms.Normalize(
                    num_img_channels * [0.5], num_img_channels * [0.5]
                ),
            ]
        )  # should this be applied the exact same way for both images?
    else:
        raise NotImplementedError("loading ldct images as np arrays not supported!")

    if num_img_channels == 1:
        PIL_image_type = "L"
    elif num_img_channels == 3:
        PIL_image_type = "RGB"
    else:
        PIL_image_type = None

    if config.segmentation_guided:
        preprocess_segmentation = transforms.Compose(
            [
                transforms.Resize(
                    (config.image_size, config.image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

        def transform(examples):
            if img_dir is not None:
                if not load_images_as_np_arrays:
                    images = [
                        preprocess(image.convert(PIL_image_type))
                        for image in examples["image"]
                    ]
                    lds_images = [
                        preprocess(image.convert(PIL_image_type))
                        for image in examples["lds_cond"]
                    ]
                else:
                    raise NotImplementedError(
                        "loading ldct images as np arrays not supported!"
                    )

            images_filenames = examples["image_filename"]
            lds_filenames = examples["lds_cond_filename"]

            segs = {}
            for seg_type in seg_types:
                segs["seg_{}".format(seg_type)] = [
                    preprocess_segmentation(image.convert("L"))
                    for image in examples["seg_{}".format(seg_type)]
                ]
            if img_dir is not None:
                return {
                    **{"images": images},
                    **{"lds_cond": lds_images},
                    **segs,
                    **{"image_filenames": images_filenames},
                    **{"lds_filenames": lds_filenames},
                }
            else:
                raise NotImplementedError(
                    "training/evaluation without images not supported!"
                )

        dataset_train.set_transform(transform)
        dataset_eval.set_transform(transform)

    else:
        # no segmentation guidance = baseline with only ldct conditioning
        if img_dir is not None:

            def transform(examples):
                if not load_images_as_np_arrays:
                    images = [
                        preprocess(image.convert(PIL_image_type))
                        for image in examples["image"]
                    ]
                    lds_images = [
                        preprocess(image.convert(PIL_image_type))
                        for image in examples["lds_cond"]
                    ]
                else:
                    raise NotImplementedError(
                        "loading ldct images as np arrays not supported!"
                    )
                images_filenames = examples["image_filename"]
                lds_filenames = examples["lds_cond_filename"]
                return {
                    "images": images,
                    "lds_cond": lds_images,
                    **{"image_filenames": images_filenames},
                    **{"lds_filenames": lds_filenames},
                }

            dataset_train.set_transform(transform)
            dataset_eval.set_transform(transform)

    if (img_dir is None) and (not segmentation_guided):
        raise NotImplementedError("training/evaluation without images not supported!")
        # train_dataloader = None
        # # just make placeholder dataloaders to iterate through when sampling from uncond model
        # eval_dataloader = torch.utils.data.DataLoader(
        #     torch.utils.data.TensorDataset(
        #         torch.zeros(
        #             config.eval_batch_size,
        #             num_img_channels,
        #             config.image_size,
        #             config.image_size,
        #         )
        #     ),
        #     batch_size=config.eval_batch_size,
        #     shuffle=eval_shuffle_dataloader,
        # )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=config.train_batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        # --comparison forces no shuffle + fixed seed for deterministic eval
        if comparison:
            eval_shuffle_dataloader = False
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=config.eval_batch_size,
            shuffle=eval_shuffle_dataloader,
            num_workers=2,
            pin_memory=True,
        )

    # define the model
    # No LDCT channel — translation uses SDEdit (start from noisy LDCT, denoise normally).
    # Seg masks are the only extra channels (if segmentation_guided).
    in_channels = num_img_channels
    if config.segmentation_guided:
        assert config.num_segmentation_classes is not None
        assert (
            config.num_segmentation_classes > 1
        ), "must have at least 2 segmentation classes (INCLUDING background)"
        if config.segmentation_channel_mode == "single":
            in_channels += 1
        elif config.segmentation_channel_mode == "multi":
            in_channels = len(seg_types) + in_channels

    model = diffusers.UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=in_channels,  # the number of input channels, 3 for RGB images (already considers the conditioning channels)
        out_channels=num_img_channels,  # the number of output channels = want to generate one single image
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # Load pretrained weights for finetuning or evaluation
    if (mode == "train" and resume_epoch is not None) or "eval" in mode:
        if mode == "train":
            print("resuming from model at training epoch {}".format(resume_epoch))
        elif "eval" in mode:
            print("loading saved model...")
        model = model.from_pretrained(
            os.path.join(config.output_dir, "unet"), use_safetensors=True
        )
    elif mode == "train" and pretrained_model_path is not None:
        print("loading pretrained model from: {}".format(pretrained_model_path))
        try:
            # Accept either:
            #   - a direct path to the UNet folder (contains config.json + weights)
            #   - a parent folder that contains a `unet/` subfolder
            resolved_pretrained_path = pretrained_model_path
            if os.path.isdir(os.path.join(pretrained_model_path, "unet")):
                candidate = os.path.join(pretrained_model_path, "unet")
                # Prefer the nested `unet/` if it looks like a diffusers component folder
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    resolved_pretrained_path = candidate

            print(f"resolved pretrained model path: {resolved_pretrained_path}")
            pretrained_model = diffusers.UNet2DModel.from_pretrained(
                resolved_pretrained_path, use_safetensors=True
            )
            
            # Check if input channels match
            pretrained_in_channels = pretrained_model.config.in_channels
            if pretrained_in_channels != in_channels:
                print(f"Channel mismatch: pretrained model has {pretrained_in_channels} input channels, "
                      f"but current config requires {in_channels} channels for translation task.")
                print("Adapting first convolutional layer to support additional conditioning channels...")
                
                # Load all weights from pretrained model
                model_state_dict = model.state_dict()
                pretrained_state_dict = pretrained_model.state_dict()
                
                # Copy all compatible weights
                for name, param in pretrained_state_dict.items():
                    # Skip the first conv layer's weight (we'll handle it separately)
                    if name == "conv_in.weight":
                        continue
                    if name in model_state_dict and param.shape == model_state_dict[name].shape:
                        model_state_dict[name] = param
                
                # Handle the first conv layer specially to accommodate new channels
                if "conv_in.weight" in pretrained_state_dict:
                    pretrained_conv_weight = pretrained_state_dict["conv_in.weight"]
                    new_conv_weight = model_state_dict["conv_in.weight"].clone()
                    
                    # Copy pretrained weights to the corresponding channels
                    # Assuming the extra channels are for additional conditioning (lds_cond, seg masks)
                    num_channels_to_copy = min(pretrained_in_channels, in_channels)
                    new_conv_weight[:, :num_channels_to_copy, :, :] = pretrained_conv_weight[:, :num_channels_to_copy, :, :]
                    
                    # Initialize new channels (for translation conditioning) with small random values
                    # or copy from existing channels as a reasonable initialization
                    if in_channels > pretrained_in_channels:
                        # Option 1: Initialize new channels with small random values
                        # new_conv_weight[:, pretrained_in_channels:, :, :] *= 0.01
                        
                        # Option 2: Copy from first channel (better for similar modalities)
                        for i in range(pretrained_in_channels, in_channels):
                            new_conv_weight[:, i:i+1, :, :] = pretrained_conv_weight[:, :1, :, :] * 0.01
                    
                    model_state_dict["conv_in.weight"] = new_conv_weight
                    print(f"Adapted conv_in layer from {pretrained_in_channels} to {in_channels} channels")
                
                model.load_state_dict(model_state_dict)
                print("Pretrained weights loaded and adapted successfully!")
            else:
                # Channels match — keep the current `model` (so we preserve `sample_size` = img_size)
                model.load_state_dict(pretrained_model.state_dict())
                print("Pretrained weights loaded successfully (channels match; kept current model config)!")
        except Exception as e:
            raise RuntimeError(
                "Failed to load pretrained model from {}: {}. "
                "Training would continue with random weights; raising so you can fix the path or architecture."
                .format(pretrained_model_path, e)
            ) from e
    else:
        if mode == "train":
            print("Training from scratch (no pretrained weights provided)")

    # Only use DataParallel if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        print(f"Using DataParallel with {num_gpus} GPUs (device_ids={device_ids}); "
              f"batch will be split: {config.train_batch_size} samples -> {config.train_batch_size // num_gpus} per GPU")
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        print(f"Using single GPU")
    model.to(device)

    # define noise scheduler
    noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)

    if mode == "train":
        # training setup — hybrid Muon (weight matrices) + AdamW (biases/norms)
        muon_params, adamw_params = get_muon_and_adamw_params(model)
        print(
            f"Optimizer split: {len(muon_params)} Muon param tensors "
            f"(lr={config.learning_rate:.2e}), "
            f"{len(adamw_params)} AdamW param tensors "
            f"(lr={config.learning_rate * 0.1:.2e})"
        )
        optimizer = HybridMuonAdamW(
            muon_params,
            adamw_params,
            muon_lr=config.learning_rate,
            adamw_lr=config.learning_rate * 0.1,  # 1e-5 when muon_lr=1e-4
        )

        updates_per_epoch = math.ceil(
            len(train_dataloader) / config.gradient_accumulation_steps
        )
        total_updates = updates_per_epoch * config.num_epochs

        # Cosine schedule is applied to Muon only; AdamW runs at its fixed lr
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer.muon,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=total_updates,
        )
        # Restore optimizer & scheduler state if resuming
        start_global_step = 0
        if resume_epoch is not None:
            ckpt_path = os.path.join(config.output_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                print(f"Loading full checkpoint from {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                start_global_step = checkpoint["global_step"]
                print(f"  Restored optimizer & scheduler at epoch {checkpoint['epoch']}, global_step {start_global_step}")
            else:
                print(f"WARNING: No checkpoint.pt found at {ckpt_path}, resuming with fresh optimizer/scheduler (model weights only)")

        # train
        train_loop(
            config,
            model,
            noise_scheduler,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
            device=device,
            start_global_step=start_global_step,
        )
    elif mode == "eval":
        """
        default eval behavior:
        evaluate image generation or translation (if for conditional model, either evaluate naive class conditioning but not CFG,
        or with CFG),
        possibly conditioned on masks.

        has various options.
        """
        # --comparison: seed everything for deterministic, reproducible eval
        eval_generator = None
        if comparison:
            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            eval_generator = torch.Generator(device=device).manual_seed(seed)
            print(f"Comparison mode: seeded all randomness with seed={seed}")

        evaluate_generation(
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            translate=True,
            eval_mask_removal=eval_mask_removal,
            eval_blank_mask=eval_blank_mask,
            device=device,
            generator=eval_generator,
        )

    elif mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(
            eval_sample_size,
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            device=device,
        )

    else:
        raise ValueError('mode "{}" not supported.'.format(mode))


if __name__ == "__main__":
    # parse args:
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_img_channels", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mayo_challenge")
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--seg_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="DDPM")
    parser.add_argument(
        "--segmentation_guided",
        action="store_true",
        help="use segmentation guided training/sampling?",
    )
    parser.add_argument(
        "--segmentation_channel_mode",
        type=str,
        default="single",
        help="single == all segmentations in one channel, multi == each segmentation in its own channel",
    )
    parser.add_argument(
        "--num_segmentation_classes",
        type=int,
        default=None,
        help="number of segmentation classes, including background",
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Muon learning rate; AdamW (biases/norms) uses 0.3x this value automatically",
    )
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="resume training starting at this epoch",
    )

    # novel options
    parser.add_argument(
        "--use_ablated_segmentations",
        action="store_true",
        help="use mask ablated training and any evaluation? sometimes randomly remove class(es) from mask during training and sampling.",
    )

    # other options
    parser.add_argument(
        "--eval_noshuffle_dataloader",
        action="store_true",
        help="if true, don't shuffle the eval dataloader",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="deterministic eval: seeds all randomness and disables shuffle so the same batch and noise are used every run",
    )

    # args only used in eval
    parser.add_argument(
        "--eval_mask_removal",
        action="store_true",
        help="if true, evaluate gradually removing anatomies from mask and re-sampling",
    )
    parser.add_argument(
        "--eval_blank_mask",
        action="store_true",
        help="if true, evaluate sampling conditioned on blank (zeros) masks",
    )
    parser.add_argument(
        "--eval_sample_size",
        type=int,
        default=1000,
        help="number of images to sample when using eval_many mode",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="number of micro-steps to accumulate before each optimizer step",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="path to pretrained model directory (containing 'unet' subfolder) for finetuning. "
             "If provided during training, will load pretrained weights and adapt them for translation task if needed.",
    )

    args = parser.parse_args()

    main(
        args.mode,
        args.img_size,
        args.num_img_channels,
        args.dataset,
        args.img_dir,
        args.seg_dir,
        args.model_type,
        args.segmentation_guided,
        args.segmentation_channel_mode,
        args.num_segmentation_classes,
        args.train_batch_size,
        args.eval_batch_size,
        args.num_epochs,
        args.learning_rate,
        args.resume_epoch,
        args.use_ablated_segmentations,
        not args.eval_noshuffle_dataloader,
        # args only used in eval
        args.eval_mask_removal,
        args.eval_blank_mask,
        args.eval_sample_size,
        args.gradient_accumulation_steps,
        args.pretrained_model_path,
        comparison=args.comparison,
        argv=sys.argv,
    )
