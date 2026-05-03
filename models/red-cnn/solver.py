import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from networks import RED_CNN
from measure import compute_measure, compute_CNR


class Solver(object):
    def __init__(self, args, data_loader, val_loader=None):
        self.args = args
        self.mode = args.mode
        self.data_loader = data_loader
        self.val_loader = val_loader

        self.device = torch.device(args.device if args.device
                                   else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu
        self.result_fig = args.result_fig
        self.patch_size = args.patch_size

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.save_freq = args.save_freq
        self.test_iters = args.test_iters
        self.resume_iter = args.resume_iter

        self.use_mask = args.use_mask
        self.mask_loss_alpha = args.mask_loss_alpha

        in_channels = 2 if self.use_mask else 1
        self.REDCNN = RED_CNN(in_channels=in_channels)
        if self.use_mask:
            # mask column starts at zero so training begins identical to baseline;
            # backprop gradually learns to use the mask without disrupting denoising
            nn.init.zeros_(self.REDCNN.conv1.weight[:, 1:, :, :])
        if self.multi_gpu and torch.cuda.device_count() > 1:
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)

        self.best_val_ssim = 0.0
        self.log_file = os.path.join(self.save_path, 'train.log')

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def _log(self, msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        with open(self.log_file, 'a') as f:
            f.write(line + '\n')

    def _save_checkpoint(self, epoch, total_iters, tag=None, is_best=False, is_latest=False):
        ckpt = {
            'epoch': epoch,
            'iteration': total_iters,
            'model_state_dict': self.REDCNN.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ssim': self.best_val_ssim,
        }
        if tag is not None:
            torch.save(ckpt, os.path.join(self.save_path, f'REDCNN_{tag}iter.ckpt'))
        if is_latest:
            torch.save(ckpt, os.path.join(self.save_path, 'latest.ckpt'))
        if is_best:
            torch.save(ckpt, os.path.join(self.save_path, 'best.ckpt'))

    def _load_checkpoint(self, iter_=None, path=None):
        if path is None:
            path = os.path.join(self.save_path, f'REDCNN_{iter_}iter.ckpt')
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            if self.multi_gpu:
                state = OrderedDict([(k[7:], v) for k, v in ckpt['model_state_dict'].items()])
                self.REDCNN.load_state_dict(state)
            else:
                self.REDCNN.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.best_val_ssim = ckpt.get('best_val_ssim', 0.0)
            return ckpt.get('epoch', 0), ckpt.get('iteration', 0)
        else:
            # legacy format: bare state_dict
            self.REDCNN.load_state_dict(ckpt)
            return 0, 0

    # kept for backwards compat with test_iters path
    def load_model(self, iter_):
        self._load_checkpoint(iter_=iter_)

    def lr_decay(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5

    def denormalize_(self, image):
        return image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(*original_result[:3]),
                         fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(*pred_result[:3]),
                         fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)
        f.savefig(os.path.join(self.save_path, 'fig', f'result_{fig_name}.png'))
        plt.close()

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def _unpack_batch_train(self, batch):
        """Return (x, y, mask_or_None) as float tensors on device, shaped (N,1,ps,ps)."""
        ps = self.patch_size
        if self.use_mask:
            x, y, mask = batch
            mask = mask.unsqueeze(0).float().to(self.device)
            mask = mask.view(-1, 1, ps, ps)
        else:
            x, y = batch[:2]
            mask = None
        x = x.unsqueeze(0).float().to(self.device).view(-1, 1, ps, ps)
        y = y.unsqueeze(0).float().to(self.device).view(-1, 1, ps, ps)
        return x, y, mask

    def _run_val(self, epoch, total_iters):
        self.REDCNN.eval()
        psnr_sum = ssim_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if self.use_mask:
                    x, y, mask, _ = batch
                    mask = mask.unsqueeze(0).float().to(self.device)
                else:
                    x, y = batch[:2]
                    mask = None
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                inp = torch.cat([x, mask], dim=1) if mask is not None else x
                pred = self.REDCNN(inp)

                x_hu = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu()))
                y_hu = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu()))
                p_hu = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu()))
                data_range = self.trunc_max - self.trunc_min
                _, pred_result = compute_measure(x_hu, y_hu, p_hu, data_range)
                psnr_sum += pred_result[0]
                ssim_sum += pred_result[1]
                n += 1

        val_psnr = psnr_sum / max(n, 1)
        val_ssim = ssim_sum / max(n, 1)
        self._log(f'Epoch {epoch}/{self.num_epochs} | Iter {total_iters} | '
                  f'Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}')

        is_best = val_ssim > self.best_val_ssim
        if is_best:
            self.best_val_ssim = val_ssim
        self._save_checkpoint(epoch, total_iters, is_best=is_best, is_latest=True)
        return val_psnr, val_ssim

    def train(self):
        start_epoch, total_iters = 1, 0
        if self.resume_iter:
            start_epoch, total_iters = self._load_checkpoint(iter_=self.resume_iter)
            start_epoch += 1
            self._log(f'Resumed from iter {self.resume_iter} (epoch {start_epoch})')

        train_losses = []
        wall_start = time.time()

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.REDCNN.train(True)

            for iter_, batch in enumerate(self.data_loader):
                total_iters += 1
                x, y, mask = self._unpack_batch_train(batch)

                inp = torch.cat([x, mask], dim=1) if mask is not None else x
                pred = self.REDCNN(inp)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                if total_iters % self.print_iters == 0:
                    self._log(f'Epoch {epoch}/{self.num_epochs} | '
                              f'Iter {total_iters} [{iter_+1}/{len(self.data_loader)}] | '
                              f'Loss: {loss.item():.8f} | '
                              f'Time: {time.time()-wall_start:.1f}s')

                if total_iters % self.decay_iters == 0:
                    self.lr_decay()

                if total_iters % self.save_iters == 0:
                    self._save_checkpoint(epoch, total_iters, tag=total_iters)
                    np.save(os.path.join(self.save_path,
                                        f'loss_{total_iters}_iter.npy'),
                            np.array(train_losses))

            # end-of-epoch checkpoint and validation
            self._save_checkpoint(epoch, total_iters, is_latest=True)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, total_iters, tag=f'epoch{epoch}')

            if self.val_loader is not None:
                self._run_val(epoch, total_iters)

    # ------------------------------------------------------------------ #
    # Testing                                                              #
    # ------------------------------------------------------------------ #

    def test(self):
        in_channels = 2 if self.use_mask else 1
        self.REDCNN = RED_CNN(in_channels=in_channels).to(self.device)
        if self.args.load_best:
            self._load_checkpoint(path=os.path.join(self.save_path, 'best.ckpt'))
        else:
            self._load_checkpoint(iter_=self.test_iters)
        self.REDCNN.eval()

        data_range = self.trunc_max - self.trunc_min
        rows = []

        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                # batch is (x, y, fname) or (x, y, mask, fname) depending on
                # whether mask_dir was supplied to the loader
                has_eval_mask = (len(batch) == 4)

                if has_eval_mask:
                    x, y, mask_tensor, fnames = batch
                    mask_np = mask_tensor[0].numpy()   # (H, W) for CNR
                else:
                    x, y, fnames = batch
                    mask_tensor = None
                    mask_np = None

                fname = fnames[0] if isinstance(fnames, (list, tuple)) else fnames
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.use_mask and has_eval_mask:
                    mask_dev = mask_tensor.unsqueeze(0).float().to(self.device)
                    inp = torch.cat([x, mask_dev], dim=1)
                else:
                    inp = x
                pred = self.REDCNN(inp)

                x_hu = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y_hu = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                p_hu = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                _, pred_result = compute_measure(x_hu, y_hu, p_hu, data_range)
                psnr, ssim, rmse_hu = pred_result

                cnr = 0.0
                if mask_np is not None:
                    # mask_np is in [0,1]; denormalize label range to use as boolean roi
                    cnr = compute_CNR(p_hu.numpy(), mask_np)

                rows.append({
                    'slice_name': fname,
                    'psnr': round(psnr, 6),
                    'ssim': round(ssim, 6),
                    'rmse_hu': round(rmse_hu, 6),
                    'cnr': round(cnr, 6),
                })

                if self.result_fig:
                    ori_result, _ = compute_measure(x_hu, y_hu, p_hu, data_range)
                    self.save_fig(x_hu, y_hu, p_hu, i, ori_result, pred_result)

                if (i + 1) % 10 == 0 or (i + 1) == len(self.data_loader):
                    print(f'\r  processed {i+1}/{len(self.data_loader)}', end='', flush=True)

        print()

        # save per-slice CSV
        csv_path = os.path.join(self.save_path, 'test_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['slice_name', 'psnr', 'ssim', 'rmse_hu', 'cnr'])
            writer.writeheader()
            writer.writerows(rows)
        print(f'Per-slice metrics saved to {csv_path}')

        # summary
        psnrs = [r['psnr'] for r in rows]
        ssims = [r['ssim'] for r in rows]
        rmses = [r['rmse_hu'] for r in rows]
        cnrs = [r['cnr'] for r in rows]
        summary = (
            f'TEST | '
            f'psnr {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f}, '
            f'ssim {np.mean(ssims):.3f} ± {np.std(ssims):.3f}, '
            f'rmse {np.mean(rmses):.2f} ± {np.std(rmses):.2f}, '
            f'cnr {np.mean(cnrs):.3f} ± {np.std(cnrs):.3f}'
        )
        self._log(summary)
