import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver


def main(args):
    cudnn.benchmark = True

    os.makedirs(args.save_path, exist_ok=True)
    if args.result_fig:
        os.makedirs(os.path.join(args.save_path, 'fig'), exist_ok=True)

    # mask_dir for model input (only when use_mask=True)
    # eval_mask_dir is always passed at test time for CNR computation
    model_mask_dir = args.mask_dir if args.use_mask else None
    eval_mask_dir = args.mask_dir if (args.mode == 'test' and args.mask_dir) else model_mask_dir

    # val loader used during training for checkpoint selection
    val_loader = None
    if args.mode == 'train':
        val_loader = get_loader(
            mode='val',
            data_path=args.data_path,
            mask_dir=model_mask_dir,
            norm_range_min=args.norm_range_min,
            norm_range_max=args.norm_range_max,
            patch_n=None,
            patch_size=None,
            batch_size=1,
            num_workers=args.num_workers,
        )

    train_patch_n = args.patch_n if args.mode == 'train' else None
    train_patch_size = args.patch_size if args.mode == 'train' else None

    data_loader = get_loader(
        mode=args.mode,
        data_path=args.data_path,
        mask_dir=eval_mask_dir,
        norm_range_min=args.norm_range_min,
        norm_range_max=args.norm_range_max,
        patch_n=train_patch_n,
        patch_size=train_patch_size,
        batch_size=(args.batch_size if args.mode == 'train' else 1),
        num_workers=args.num_workers,
    )

    solver = Solver(args, data_loader, val_loader=val_loader)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/png_dataset/',
                        help='Root of png_dataset/ containing {train,val,test}/1mm/{QD,FD}/')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--result_fig', action='store_true', default=False)

    # mask conditioning
    parser.add_argument('--use_mask', action='store_true', default=False,
                        help='Use segmentation mask as loss weight during training')
    parser.add_argument('--mask_dir', type=str, default='./dataset/multilabel/',
                        help='Root of multilabel/ containing {tissue}/{split}/1mm/QD/')
    parser.add_argument('--mask_loss_alpha', type=float, default=1.0,
                        help='Extra loss weight on tissue pixels: weight = 1 + alpha * (mask > 0)')

    # normalisation (must match how PNGs were saved)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    # patch training
    parser.add_argument('--patch_n', type=int, default=10,
                        help='Patches sampled per image per iteration')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    # optimisation
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)

    # checkpointing
    parser.add_argument('--save_iters', type=int, default=1000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save named epoch checkpoint every N epochs')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Resume from REDCNN_{N}iter.ckpt (0 = start fresh)')
    parser.add_argument('--test_iters', type=int, default=1000,
                        help='Iteration checkpoint to load for testing')
    parser.add_argument('--load_best', action='store_true', default=False,
                        help='Load best.ckpt instead of REDCNN_{N}iter.ckpt for testing')

    # hardware
    parser.add_argument('--device', type=str, default='',
                        help='e.g. cuda:0  (empty = auto-detect)')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--multi_gpu', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
