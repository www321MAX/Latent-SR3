"""
两阶段训练：
  Stage 1: 预训练 VAE（MSE重建 + KL散度 + LPIPS感知损失）
  Stage 2: 训练扩散 UNet（Min-SNR-γ 加权噪声预测）

数据集：DIV2K（800张）+ Flickr2K（2650张）混合，共 3450 对图像
  主训练集放在 --train_dir（含 LR/ 和 HR/ 子目录）
  Flickr2K 等额外数据集通过 --extra_train_dirs 追加（逗号分隔）

目录结构示例：
  data/
    train/div2k/LR/  train/div2k/HR/
    train/flickr2k/LR/     train/flickr2k/HR/
    valid/LR/        valid/HR/

用法:
  # 两阶段完整训练
  python train_sr3.py --stage both --fp16
      --train_dir ./data/train/div2k
      --extra_train_dirs ./data/train/flickr2k
      --valid_dir ./data/valid

  # 只训练扩散模型（已有VAE权重）
  python train_sr3.py --stage diff --fp16
      --vae_ckpt ./checkpoints/vae_best.pt

  # 推理
  python train_sr3.py --infer
      --lr_img ./test_lr.png
      --ckpt ./checkpoints/sr3_best.pt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决部分环境下的 OpenMP 冲突问题

import sys
import argparse
import math
import time
import glob
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model.latent_sr3 import (
    LatentSR3, VAEEncoder, VAEDecoder, SRDataset,
    LPIPSLoss, compute_psnr, compute_ssim,
)


#配置

def get_config():
    p = argparse.ArgumentParser(
        description='Latent SR3 Training — DIV2K + Flickr2K 混合配置'
    )

    #模式
    p.add_argument('--stage', type=str, default='both',
                   choices=['vae', 'diff', 'both'],
                   help='训练阶段: vae / diff / both（默认 both）')
    p.add_argument('--infer', action='store_true',
                   help='推理模式，需同时指定 --ckpt 和 --lr_img')

    #数据路径
    p.add_argument('--train_dir', type=str, default='./data/train/Div2K',
                   help='主训练集根目录（含 LR/ HR/ 子目录），默认 DIV2K train')
    p.add_argument('--extra_train_dirs', type=str,
                   default='./data/train/Flickr2K',
                   help='额外训练集目录，逗号分隔；默认追加 Flickr2K')
    p.add_argument('--valid_dir', type=str, default='./data/valid',
                   help='验证集根目录（含 LR/ HR/ 子目录）')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--pin_memory', type=lambda x: x.lower()!='false',
                   default=True,
                   help='DataLoader pin_memory（大显存卡建议False，减少分配器预留，默认True）')

    #模型结构
    p.add_argument('--lr_size',        type=int, default=64)
    p.add_argument('--hr_size',        type=int, default=512)
    p.add_argument('--latent_ch',      type=int, default=8)
    p.add_argument('--vae_base_ch',    type=int, default=160)
    p.add_argument('--vae_ch_mult',  type=str, default='1,2,4',
               help='VAE 各层通道倍率，逗号分隔整数（默认 1,2,4 即 8× 下采样)')
    p.add_argument('--unet_ch_mult', type=str, default='1,2,4,8',
                help='UNet 各层通道倍率，逗号分隔整数（默认 1,2,4,8）')
    p.add_argument('--unet_base_ch',   type=int, default=192)
    p.add_argument('--num_timesteps',  type=int, default=1000)
    p.add_argument('--num_res_blocks', type=int, default=2)

    #VAE 训练超参
    p.add_argument('--vae_epochs',         type=int,   default=60,
                   help='VAE 训练轮数（默认 60）')
    p.add_argument('--vae_batch',          type=int,   default=8,
                   help='VAE 批大小（默认 8）')
    p.add_argument('--vae_num_res_blocks', type=int,   default=3,
                   help='VAE 每个分辨率级别的 ResBlock 数（默认 3；改 4 加容量不改 latent 尺寸）')
    p.add_argument('--vae_lr',             type=float, default=3e-5,
                   help='VAE 学习率（默认 3e-5）')
    p.add_argument('--vae_kl_weight',      type=float, default=1e-3,
                   help='KL散度权重 β（默认 1e-3）')
    p.add_argument('--vae_lpips_weight',   type=float, default=0.1,
                   help='LPIPS感知损失权重（默认 0.1；设 0 禁用）')
    p.add_argument('--vae_lpips_every',    type=int,   default=20,
                   help='每 N 个 batch 计算一次 LPIPS（默认 20；设 1 每步都算但很慢；'
                        'CPU offload 模式下建议 ≥ 20）')
    p.add_argument('--vae_ckpt',           type=str,   default=None,
                   help='已有VAE权重路径，Stage2可直接加载跳过Stage1')

    #扩散模型训练超参
    p.add_argument('--diff_epochs',  type=int,   default=600,
                   help='扩散模型训练轮数（默认 600）')
    p.add_argument('--diff_batch',   type=int,   default=12,
                   help='扩散模型批大小（默认 12）')
    p.add_argument('--diff_lr',      type=float, default=1e-4,
                   help='UNet 学习率（默认 1e-4）')
    p.add_argument('--warmup_steps', type=int,   default=2000,
                   help='学习率预热步数（默认 2000）')
    p.add_argument('--grad_clip',    type=float, default=1.0,
                   help='梯度裁剪范数（默认 1.0）')
    p.add_argument('--ema_decay',    type=float, default=0.9999,
                   help='EMA 衰减率（默认 0.9999）')
    p.add_argument('--snr_gamma',    type=float, default=5.0,
                   help='Min-SNR-γ 上限（默认 5.0；设极大值禁用）')

    #数据增强
    p.add_argument('--aug_hflip',        type=lambda x: x.lower()!='false',
                   default=True,  help='随机水平翻转（默认 True）')
    p.add_argument('--aug_vflip',        type=lambda x: x.lower()!='false',
                   default=True,  help='随机垂直翻转（默认 True）')
    p.add_argument('--aug_rotate90',     type=lambda x: x.lower()!='false',
                   default=True,  help='随机90°旋转（默认 True）')
    p.add_argument('--aug_color_jitter', type=lambda x: x.lower()!='false',
                   default=True,  help='颜色抖动（默认 True）')
    p.add_argument('--aug_random_crop',  type=lambda x: x.lower()!='false',
                   default=True,  help='随机裁剪（默认 True，强烈建议开启）')
    p.add_argument('--aug_blur',         type=lambda x: x.lower()!='false',
                   default=True,  help='LR随机模糊（默认 True）')
    p.add_argument('--aug_noise',        type=lambda x: x.lower()!='false',
                   default=True,  help='LR随机噪声（默认 True）')
    p.add_argument('--aug_mixup',        type=lambda x: x.lower()!='false',
                   default=False, help='MixUp（DIV2K/Flickr2K 默认 False）')
    p.add_argument('--aug_cutmix',       type=lambda x: x.lower()!='false',
                   default=False, help='CutMix（默认 False）')
    p.add_argument('--crop_size',        type=int, default=512,
                   help='RandomCrop 裁剪尺寸（默认 512，与 hr_size 一致）')
    p.add_argument('--cache_mode', type=str, default='none',
                   choices=['none', 'lr_only', 'full', 'patch_disk'],
                   help=(
                       'none      : 不缓存（默认）\n'
                       'lr_only   : 只缓存 LR 图到 RAM（约 400 MB，推荐）\n'
                       'full      : 全量缓存 LR+HR 到 RAM（需 ≥ 32 GB，DIV2K+Flickr2K 约 25 GB）\n'
                       'patch_disk: 预提取 patch 到磁盘临时目录（一次性IO，之后读小文件）'
                   ))
    p.add_argument('--patch_disk_dir', type=str, default='./cache_patches',
                   help='patch_disk 模式的缓存目录（默认 ./cache_patches）')
    p.add_argument('--patches_per_img', type=int, default=10,
                   help='patch_disk 模式每张原图提取的 patch 数（默认 10）')
    p.add_argument('--vae_patch_size',   type=int, default=256,
                   help='VAE训练专用分辨率（0=使用hr_size）。'
                        '设为 256 可将显存从 ~12GB 降到 ~3GB。'
                        'VAE 在 patch 上训练，重建质量几乎不受影响（SD同款方案）。')

    #早停
    p.add_argument('--early_stop_patience',  type=int,   default=50,
                   help='验证loss连续N个epoch无改善则停止（0=禁用）')
    p.add_argument('--early_stop_min_delta', type=float, default=1e-7)
    p.add_argument('--nan_skip_limit',       type=int,   default=5,
                   help='连续NaN batch超限则中止训练')

    #推理
    p.add_argument('--ckpt',       type=str, default=None,
                   help='推理用模型权重路径')
    p.add_argument('--lr_img',     type=str, default=None,
                   help='推理输入 LR 图片路径')
    p.add_argument('--out',        type=str, default='sr_output.png',
                   help='推理结果保存路径')
    p.add_argument('--ddim_steps', type=int, default=200,
                   help='推理 DDIM 步数（默认 200）')

    #通用
    p.add_argument('--save_dir',             type=str, default='./checkpoints',
                   help='模型权重保存目录')
    p.add_argument('--save_samples_vae_dir', type=str, default='./samples_vae',
                   help='VAE 重建样本目录（Stage1 专用）')
    p.add_argument('--save_samples_sr3_dir', type=str, default='./samples_sr3',
                   help='SR 扩散样本目录（Stage2 专用）')
    p.add_argument('--log_dir',      type=str, default='./runs',
                   help='TensorBoard 日志目录')
    p.add_argument('--save_every',   type=int, default=20,
                   help='每 N 个 epoch 保存一次 checkpoint')
    p.add_argument('--log_every',    type=int, default=50,
                   help='每 N 步打印一次 loss')
    p.add_argument('--sample_every', type=int, default=50,
                   help='每 N 个 epoch 生成一次样本图')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--resume', type=str, default=None,
                   help='从 checkpoint 恢复扩散模型训练')
    p.add_argument('--fp16',   action='store_true',
                   help='启用混合精度训练（推荐，节省显存）')

    #显存优化
    p.add_argument('--gradient_checkpointing', action='store_true',
                   help='启用 UNet 梯度检查点（节省约 30-50%% 显存，略微增加计算时间）')
    p.add_argument('--vae_gradient_checkpointing', action='store_true',
                   help='启用 VAE Encoder/Decoder 梯度检查点')
    p.add_argument('--lazy_ema', action='store_true',
                   help='启用 Lazy EMA：预热期间不复制权重，减少显存占用')
    p.add_argument('--lazy_ema_warmup', type=int, default=2000,
                   help='Lazy EMA 预热步数，预热期间 EMA 不更新（默认 2000）')
    p.add_argument('--clear_cache_every', type=int, default=100,
                   help='每 N 步调用 torch.cuda.empty_cache() 清理显存缓存（默认 100，0=禁用）')
    p.add_argument('--vae_accum_steps', type=int, default=1,
                   help='VAE 梯度累积步数（默认 1；显存紧张可设 2~4 来降低等效 batch 占用）')
    p.add_argument('--diff_accum_steps', type=int, default=4,
                   help='扩散模型梯度累积步数（默认 4）')
    p.add_argument('--max_alloc_mb', type=int, default=0,
                   help='设置 PYTORCH_CUDA_ALLOC_CONF max_split_size_mb 减少显存碎片'
                        '（推荐 128~512，0=不设置）')
    
    cfg = p.parse_args()
    cfg.vae_ch_mult  = tuple(int(x) for x in cfg.vae_ch_mult.split(','))
    cfg.unet_ch_mult = tuple(int(x) for x in cfg.unet_ch_mult.split(','))

    return cfg


#  早停器
class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-7,
                 nan_limit: int = 5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.nan_limit  = nan_limit
        self.best_loss  = float('inf')
        self.counter    = 0
        self.nan_count  = 0
        self.best_epoch = 0

    def step_loss(self, val_loss: float, epoch: int) -> bool:
        if self.patience == 0:
            return False
        if math.isnan(val_loss) or math.isinf(val_loss):
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"\n  早停触发！最佳epoch={self.best_epoch}, "
                  f"最佳val_loss={self.best_loss:.6f}")
            return True
        return False

    def step_nan(self) -> bool:
        self.nan_count += 1
        if self.nan_count >= self.nan_limit:
            print(f"\n  早停触发！连续 {self.nan_limit} 个batch出现NaN loss。")
            return True
        return False

    def reset_nan(self):
        self.nan_count = 0

    def state_dict(self):
        return {'best_loss': self.best_loss, 'counter': self.counter,
                'nan_count': self.nan_count, 'best_epoch': self.best_epoch}

    def load_state_dict(self, d):
        self.best_loss  = d.get('best_loss',  float('inf'))
        self.counter    = d.get('counter',    0)
        self.nan_count  = d.get('nan_count',  0)
        self.best_epoch = d.get('best_epoch', 0)


#  增强数据集
class PairedSRDataset(Dataset):
    def __init__(
        self,
        pairs: list,
        hr_size: int = 512,
        lr_size: int = 64,
        augment: bool = True,
        # ── 增强开关 ──
        aug_hflip:        bool = True,
        aug_vflip:        bool = True,
        aug_rotate90:     bool = True,
        aug_color_jitter: bool = True,
        aug_random_crop:  bool = True,
        aug_blur:         bool = True,   # 只施加于 LR
        aug_noise:        bool = True,   # 只施加于 LR
        aug_mixup:        bool = False,
        aug_cutmix:       bool = False,
        crop_size:        int  = 512,    # HR 裁剪尺寸，LR 对应 crop_size//8
        cache_in_memory:  bool = False,  # 预缓存所有图片到 RAM，消除磁盘 IO 瓶颈
    ):
        self.pairs    = pairs
        self.hr_size  = hr_size
        self.lr_size  = lr_size
        self.augment  = augment
        self.crop_size = crop_size

        self._cache: dict      = {}
        self._cache_mode       = cache_in_memory   # 复用参数名，实际存 mode 字符串
        self._cache_enabled    = (cache_in_memory not in ('none', False, None, ''))

        self.aug_hflip        = aug_hflip
        self.aug_vflip        = aug_vflip
        self.aug_rotate90     = aug_rotate90
        self.aug_color_jitter = aug_color_jitter
        self.aug_random_crop  = aug_random_crop
        self.aug_blur         = aug_blur
        self.aug_noise        = aug_noise
        self.aug_mixup        = aug_mixup
        self.aug_cutmix       = aug_cutmix

        self.hr_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # LR 最终 transform
        self.lr_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # 无增强时的 resize-only transform
        self.hr_resize = transforms.Compose([
            transforms.Resize((self.hr_size, self.hr_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.lr_resize = transforms.Compose([
            transforms.Resize((self.lr_size, self.lr_size),
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        
        if self._cache_enabled and cache_in_memory in ('lr_only', 'full'):
            self._prefetch_cache()


    def _prefetch_cache(self):
        """
        按 cache_mode 预载数据：
          lr_only   : LR 原图 + HR 预缩略图
          full      : LR+HR 原图全量载入 RAM
        """
        import time as _time
        mode = self._cache_mode
        t0   = _time.time()
        n    = len(self.pairs)
        hr_thumb_size = min(self.crop_size * 2, 1024)
        print(f"  [缓存:{mode}] 预载 {n} 对图像"
              + (f"（HR 缩略 {hr_thumb_size}px）" if mode == 'lr_only' else "")
              + "...", end='', flush=True)
        for i, (lr_path, hr_path) in enumerate(self.pairs):
            try:
                lr_img = Image.open(lr_path).convert('RGB')
                lr_img.load()
                if mode == 'full':
                    hr_img = Image.open(hr_path).convert('RGB')
                    hr_img.load()
                    self._cache[i] = (lr_img, hr_img)
                else:   # lr_only
                    hr_img = Image.open(hr_path).convert('RGB')
                    w, h   = hr_img.size
                    # 等比缩到长边 = hr_thumb_size
                    if max(w, h) > hr_thumb_size:
                        scale  = hr_thumb_size / max(w, h)
                        hr_img = hr_img.resize(
                            (int(w * scale), int(h * scale)), Image.BILINEAR)
                    hr_img.load()
                    self._cache[i] = (lr_img, hr_img)   # 存缩略图
            except Exception as e:
                print(f"\n  ⚠ 缓存失败[{i}]: {e}")
        elapsed = _time.time() - t0
        # 估算内存占用
        if self._cache:
            lr_s = sum(c[0].size[0]*c[0].size[1]*3 for c in self._cache.values())
            hr_s = sum(c[1].size[0]*c[1].size[1]*3 for c in self._cache.values() if c[1])
            total_mb = (lr_s + hr_s) / 1024**2
            print(f" 完成（{len(self._cache)}/{n}，耗时 {elapsed:.1f}s，"
                  f"占用 ~{total_mb:.0f} MB）")
        else:
            print(f" 完成（耗时 {elapsed:.1f}s）")

        
    #内部工具

    def _load_pair(self, idx):
        """读取一对 LR/HR PIL 图像；按缓存模式从 RAM 或磁盘取。"""
        lr_path, hr_path = self.pairs[idx]
        if self._cache_enabled and idx in self._cache:
            cached_lr, cached_hr = self._cache[idx]
            # full 模式：HR 是原图；lr_only 模式：HR 是缩略
            return cached_lr.copy(), cached_hr.copy()
        try:
            lr_img = Image.open(lr_path).convert('RGB')
            hr_img = Image.open(hr_path).convert('RGB')
        except Exception as e:
            print(f"   读取失败 [{lr_path}]: {e}，使用空白图替代")
            lr_img = Image.new('RGB', (self.lr_size, self.lr_size))
            hr_img = Image.new('RGB', (self.hr_size, self.hr_size))
        return lr_img, hr_img

    def _sync_random_crop(self, hr_img: Image.Image, lr_img: Image.Image):
        scale = self.hr_size // self.lr_size          
        lr_crop_size = self.crop_size // scale         

        # 确保 HR 足够大
        w, h = hr_img.size
        if w < self.crop_size or h < self.crop_size:
            hr_img = hr_img.resize(
                (max(w, self.crop_size), max(h, self.crop_size)), Image.BICUBIC)
            w, h = hr_img.size

        # HR 随机裁剪
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        hr_crop = hr_img.crop((x, y, x + self.crop_size, y + self.crop_size))

        # 按坐标比例映射到 LR
        lw, lh = lr_img.size
        if lw < lr_crop_size or lh < lr_crop_size:
            lr_img = lr_img.resize(
                (max(lw, lr_crop_size), max(lh, lr_crop_size)), Image.BICUBIC)
            lw, lh = lr_img.size

        lx = int(x / w * lw)
        ly = int(y / h * lh)
        lx = min(lx, lw - lr_crop_size)
        ly = min(ly, lh - lr_crop_size)
        lr_crop = lr_img.crop((lx, ly, lx + lr_crop_size, ly + lr_crop_size))

        # 若裁剪尺寸已等于训练尺寸则直接返回
        if hr_crop.size == (self.hr_size, self.hr_size):
            hr_out = hr_crop
        else:
            hr_out = hr_crop.resize((self.hr_size, self.hr_size), Image.BICUBIC)
        if lr_crop.size == (self.lr_size, self.lr_size):
            lr_out = lr_crop
        else:
            lr_out = lr_crop.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        return hr_out, lr_out

    #MixUp / CutMix

    def _mixup(self, hr1, lr1, hr2, lr2, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        hr  = lam * hr1 + (1 - lam) * hr2
        lr  = lam * lr1 + (1 - lam) * lr2
        return hr, lr

    def _cutmix(self, hr1, lr1, hr2, lr2, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        _, H, W = hr1.shape

        # 计算裁取框（HR 尺寸）
        cut_ratio = math.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)

        hr = hr1.clone()
        hr[:, y1:y2, x1:x2] = hr2[:, y1:y2, x1:x2]

        # LR 对应区域
        scale = W // lr1.shape[-1]
        lx1, lx2 = x1 // scale, x2 // scale
        ly1, ly2 = y1 // scale, y2 // scale
        lr = lr1.clone()
        lr[:, ly1:ly2, lx1:lx2] = lr2[:, ly1:ly2, lx1:lx2]

        return hr, lr

    def _load_and_transform(self, idx) -> tuple:
        lr_img, hr_img = self._load_pair(idx)

        if not self.augment:
            w, h = hr_img.size
            if w < self.crop_size or h < self.crop_size:
                hr_img = hr_img.resize((max(w, self.crop_size), max(h, self.crop_size)), Image.BICUBIC)
                w, h = hr_img.size
            x = (w - self.crop_size) // 2
            y = (h - self.crop_size) // 2
            hr_img = hr_img.crop((x, y, x + self.crop_size, y + self.crop_size))
            lw, lh = lr_img.size
            lr_crop = self.crop_size // (self.hr_size // self.lr_size)
            lx = (lw - lr_crop) // 2
            ly = (lh - lr_crop) // 2
            lr_img = lr_img.crop((lx, ly, lx + lr_crop, ly + lr_crop))

        if self.aug_random_crop:
            hr_img, lr_img = self._sync_random_crop(hr_img, lr_img)
        else:
            hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BILINEAR)
            lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.BILINEAR)

        if self.aug_hflip and random.random() > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.aug_vflip and random.random() > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
            lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
        if self.aug_rotate90:
            k = random.randint(0, 3)
            ops = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
            if ops[k]:
                hr_img = hr_img.transpose(ops[k])
                lr_img = lr_img.transpose(ops[k])

        hr_t = self.hr_to_tensor(hr_img)
        lr_t = self.lr_to_tensor(lr_img)

        if self.aug_color_jitter and random.random() > 0.5:
            hr_t, lr_t = self._tensor_color_jitter(hr_t, lr_t)
        if self.aug_noise and random.random() < 0.3:
            sigma = random.uniform(1.0, 6.0) / 255.0 * 2.0
            lr_t = (lr_t + torch.randn_like(lr_t) * sigma).clamp(-1, 1)
        if self.aug_blur and random.random() < 0.4:
            lr_t = self._tensor_gaussian_blur(lr_t)

        return lr_t, hr_t


    def __getitem__(self, idx):
        lr_t, hr_t = self._load_and_transform(idx)

        if (self.aug_mixup or self.aug_cutmix) and self.augment and random.random() > 0.5:
            idx2 = random.randint(0, len(self.pairs) - 1)
            lr2_t, hr2_t = self._load_and_transform(idx2)  
            if self.aug_mixup:
                hr_t, lr_t = self._mixup(hr_t, lr_t, hr2_t, lr2_t)
            else:
                hr_t, lr_t = self._cutmix(hr_t, lr_t, hr2_t, lr2_t)

        return lr_t, hr_t

    def _tensor_color_jitter(self, hr_t: torch.Tensor, lr_t: torch.Tensor):
        if random.random() > 0.5:
            factor = random.uniform(-0.2, 0.2)
            hr_t = (hr_t + factor).clamp(-1, 1)
            lr_t = (lr_t + factor).clamp(-1, 1)

        # contrast: 向均值收缩/扩张
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            hr_mean = hr_t.mean(dim=[1, 2], keepdim=True)
            lr_mean = lr_t.mean(dim=[1, 2], keepdim=True)
            hr_t = ((hr_t - hr_mean) * factor + hr_mean).clamp(-1, 1)
            lr_t = ((lr_t - lr_mean) * factor + lr_mean).clamp(-1, 1)

        # saturation: 向灰度图收缩/扩张
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            w = torch.tensor([0.299, 0.587, 0.114],
                              device=hr_t.device).view(3, 1, 1)
            hr_gray = (hr_t * w).sum(0, keepdim=True)
            lr_gray = (lr_t * w).sum(0, keepdim=True)
            hr_t = ((hr_t - hr_gray) * factor + hr_gray).clamp(-1, 1)
            lr_t = ((lr_t - lr_gray) * factor + lr_gray).clamp(-1, 1)

        # hue: RGB -> 旋转色相 -> RGB
        if random.random() > 0.7:
            angle = random.uniform(-0.05, 0.05) * math.pi * 2   
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            mat = torch.tensor([
                [cos_a + (1-cos_a)/3,       (1-cos_a)/3 - sin_a/math.sqrt(3),  (1-cos_a)/3 + sin_a/math.sqrt(3)],
                [(1-cos_a)/3 + sin_a/math.sqrt(3), cos_a + (1-cos_a)/3,        (1-cos_a)/3 - sin_a/math.sqrt(3)],
                [(1-cos_a)/3 - sin_a/math.sqrt(3), (1-cos_a)/3 + sin_a/math.sqrt(3), cos_a + (1-cos_a)/3],
            ], device=hr_t.device, dtype=hr_t.dtype)  

            def _apply_hue(t):
                C, H, W = t.shape
                t_flat = t.reshape(3, -1)          # (3, H*W)
                return (mat @ t_flat).reshape(C, H, W).clamp(-1, 1)

            hr_t = _apply_hue(hr_t)
            lr_t = _apply_hue(lr_t)

        return hr_t, lr_t

    @staticmethod
    def _tensor_gaussian_blur(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        # box filter 权重
        k = torch.ones(1, 1, kernel_size, 1, device=x.device) / kernel_size
        C = x.shape[0]
        x = x.unsqueeze(0)   # (1,C,H,W)
        k_h = k.expand(C, 1, kernel_size, 1)
        k_w = k.permute(0, 1, 3, 2).expand(C, 1, 1, kernel_size)
        pad = kernel_size // 2
        x = torch.nn.functional.conv2d(x, k_h, padding=(pad, 0), groups=C)
        x = torch.nn.functional.conv2d(x, k_w, padding=(0, pad), groups=C)
        return x.squeeze(0).clamp(-1, 1)

    def __len__(self):
        return len(self.pairs)


#数据集构建工具

def collect_pairs(root: str) -> list:
    lr_dir = os.path.join(root, 'LR')
    hr_dir = os.path.join(root, 'HR')
    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"找不到LR目录: {lr_dir}")
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"找不到HR目录: {hr_dir}")

    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
    if not lr_files:
        raise FileNotFoundError(f"在 {lr_dir} 中未找到任何 PNG 文件")

    pairs, missing = [], []
    for lr_path in lr_files:
        fname   = os.path.basename(lr_path)
        hr_path = os.path.join(hr_dir, fname)
        if os.path.isfile(hr_path):
            pairs.append((lr_path, hr_path))
        else:
            missing.append(fname)

    if missing:
        print(f"  {len(missing)} 个LR缺少配对HR（已跳过）")

    valid_pairs = []
    for lr_path, hr_path in pairs:
        try:
            with Image.open(lr_path) as img:
                lr_w, lr_h = img.size
            with Image.open(hr_path) as img:
                hr_w, hr_h = img.size
            # 检查比例是否正确（允许±1像素误差）
            if abs(hr_w / lr_w - 8) > 0.5 or abs(hr_h / lr_h - 8) > 0.5:
                print(f"  比例异常跳过: {os.path.basename(lr_path)} "
                      f"LR={lr_w}×{lr_h} HR={hr_w}×{hr_h}")
                continue
            valid_pairs.append((lr_path, hr_path))
        except Exception:
            continue
    print(f"  -> 有效配对: {len(valid_pairs)} / {len(pairs)}")
    return valid_pairs


def build_datasets(cfg):
    print("\n[数据加载] ")

    #主训练集
    print(f"  扫描主训练集: {cfg.train_dir}")
    train_pairs = collect_pairs(cfg.train_dir)

    #额外训练集
    if cfg.extra_train_dirs:
        for extra_dir in cfg.extra_train_dirs.split(','):
            extra_dir = extra_dir.strip()
            if extra_dir:
                print(f"  扫描额外训练集: {extra_dir}")
                extra_pairs = collect_pairs(extra_dir)
                train_pairs += extra_pairs
                print(f"    -> 追加 {len(extra_pairs)} 对")

    #验证集
    print(f"  扫描验证集: {cfg.valid_dir}")
    valid_pairs = collect_pairs(cfg.valid_dir)

    print(f"  训练总对数: {len(train_pairs)}  验证对数: {len(valid_pairs)}")

    aug_kwargs = dict(
        aug_hflip        = cfg.aug_hflip,
        aug_vflip        = cfg.aug_vflip,
        aug_rotate90     = cfg.aug_rotate90,
        aug_color_jitter = cfg.aug_color_jitter,
        aug_random_crop  = cfg.aug_random_crop,
        aug_blur         = cfg.aug_blur,
        aug_noise        = cfg.aug_noise,
        aug_mixup        = cfg.aug_mixup,
        aug_cutmix       = cfg.aug_cutmix,
        crop_size        = cfg.crop_size,
    )

    cache = getattr(cfg, 'cache_mode', 'none')
    train_ds = PairedSRDataset(
        train_pairs, cfg.hr_size, cfg.lr_size,
        augment=True, cache_in_memory=cache, **aug_kwargs
    )
    val_ds = PairedSRDataset(
        valid_pairs, cfg.hr_size, cfg.lr_size,
        augment=False, 
        aug_random_crop= False,
        cache_in_memory=cache
    )
    return train_ds, val_ds


#EMA

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 lazy_ema: bool = False, lazy_ema_warmup: int = 2000):
        self.decay           = decay
        self.lazy_ema        = lazy_ema
        self.lazy_ema_warmup = lazy_ema_warmup
        self._step           = 0
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._step += 1
        # 预热期间跳过权重复制，仅更新步数计数
        if self.lazy_ema and self._step < self.lazy_ema_warmup:
            return
        for k, v in model.state_dict().items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"   EMA跳过更新：参数 {k} 含有NaN/Inf")
                continue
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def restore(self, original_sd: dict, model: nn.Module):
        model.load_state_dict(original_sd)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = sd


#STAGE 1: VAE 预训练

class VAETrainer:
    """
    Stage 1: VAE 预训练。
    损失 = MSE重建 + β·KL + λ_lpips·LPIPS感知损失
    """
    def __init__(self, cfg, train_loader, val_loader, writer):
        self.cfg          = cfg
        self.device       = cfg.device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.writer       = writer

        use_ckpt = getattr(cfg, 'vae_gradient_checkpointing', False)
        vae_ch_mult = getattr(cfg, 'vae_ch_mult', (1, 2, 4))
        self.encoder = VAEEncoder(
            in_ch=3, latent_ch=cfg.latent_ch,
            base_ch=cfg.vae_base_ch,
            ch_mult=vae_ch_mult,
            num_res_blocks=getattr(cfg, 'vae_num_res_blocks', 2),
            use_checkpoint=use_ckpt,
        ).to(self.device)
        self.decoder = VAEDecoder(
            out_ch=3, latent_ch=cfg.latent_ch,
            base_ch=cfg.vae_base_ch,
            ch_mult=tuple(reversed(vae_ch_mult)),
            num_res_blocks=getattr(cfg, 'vae_num_res_blocks', 2),
            use_checkpoint=use_ckpt,
        ).to(self.device)

        #LPIPS 感知损失
        self.lpips_weight = getattr(cfg, 'vae_lpips_weight', 0.0) or 0.0
        if self.lpips_weight > 0:
            self.lpips_fn = LPIPSLoss().to(self.device).eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
            print(f"  LPIPS 感知损失已启用（GPU），权重={self.lpips_weight}")
        else:
            self.lpips_fn = None
            print("  LPIPS 感知损失已禁用（vae_lpips_weight=0）")

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = AdamW(params, lr=cfg.vae_lr, weight_decay=1e-4)
        total_steps = cfg.vae_epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        self.scaler       = torch.amp.GradScaler(enabled=(cfg.fp16 and torch.cuda.is_available()))
        self.best_val     = float('inf')
        self.global_step  = 0
        self.early_stopping = EarlyStopping(
            patience  = cfg.early_stop_patience,
            min_delta = cfg.early_stop_min_delta,
            nan_limit = cfg.nan_skip_limit,
        )
        #分阶段样本目录
        self.sample_dir = getattr(cfg, 'save_samples_vae_dir', './samples_vae')
        os.makedirs(cfg.save_dir,    exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        if cfg.vae_ckpt and os.path.exists(cfg.vae_ckpt):
            self._load(cfg.vae_ckpt)

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        print(f"  已加载VAE权重: {path}")

    def _save(self, tag='best'):
        path = os.path.join(self.cfg.save_dir, f'vae_{tag}.pt')
        torch.save({'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict()}, path)
        return path

    def vae_loss(self, x):
        mean, logvar = self.encoder(x)
        mean   = mean.clamp(-10, 10)
        logvar = logvar.clamp(-10, 10)
        std    = torch.exp(0.5 * logvar)
        z      = mean + std * torch.randn_like(std)
        recon  = self.decoder(z)

        l_recon  = F.mse_loss(recon, x)
        mean_f   = mean.float()
        logvar_f = logvar.float()
        l_kl     = -0.5 * torch.mean(1 + logvar_f - mean_f.pow(2) - logvar_f.exp())

        loss = l_recon + self.cfg.vae_kl_weight * l_kl
        return loss, l_recon, l_kl, recon

    def train(self):
        print("\n" + "═" * 55)
        print("  STAGE 1: VAE 预训练")
        vae_patch = getattr(self.cfg, 'vae_patch_size', 0)
        eff_size  = vae_patch if vae_patch > 0 else self.cfg.hr_size
        print(f"  epochs={self.cfg.vae_epochs}  lr={self.cfg.vae_lr}"
              f"  kl_weight={self.cfg.vae_kl_weight}"
              f"  lpips_weight={self.lpips_weight}")
        print(f"  有效输入分辨率: {eff_size}×{eff_size}  "
              f"(vae_patch_size={'disabled' if vae_patch==0 else vae_patch})")
        if torch.cuda.is_available():
            alloc  = torch.cuda.memory_allocated()  / 1024**3
            reserv = torch.cuda.memory_reserved()   / 1024**3
            print(f"  [显存] 初始化后: allocated={alloc:.2f}GiB  reserved={reserv:.2f}GiB")
        print("═" * 55)

        for epoch in range(1, self.cfg.vae_epochs + 1):
            self.encoder.train(); self.decoder.train()
            ep_loss = ep_recon = ep_kl = ep_lpips = 0.0
            t0 = time.time()
            skipped = 0

            vae_accum   = getattr(self.cfg, 'vae_accum_steps', 1)
            lpips_every = getattr(self.cfg, 'vae_lpips_every', 10)
            data_time = gpu_time = 0.0   
            t_data = time.time()
            for batch_idx, (_, hr) in enumerate(self.train_loader):
                if batch_idx < 5:
                    data_time += time.time() - t_data
                t_gpu = time.time()
                try:
                    hr = hr.to(self.device, non_blocking=True)
                    do_lpips = (self.lpips_fn is not None
                                and lpips_every > 0
                                and self.global_step % lpips_every == 0)
                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

                    # autocast 只包 VAE forward，不包 LPIPS
                    with torch.amp.autocast(device_type, enabled=self.cfg.fp16):
                        loss, l_recon, l_kl, recon = self.vae_loss(hr)

                    # LPIPS 在 autocast 外，全 float32
                    l_lpips = torch.tensor(0.0, device=self.device)
                    if do_lpips:
                        l_lpips = self.lpips_fn(
                            recon.float(),
                            hr.float(),
                        )
                        loss = loss + self.lpips_weight * l_lpips.to(loss.dtype)

                    del recon, hr

                    if not torch.isfinite(loss):
                        skipped += 1
                        del loss, l_recon, l_kl, l_lpips
                        if self.early_stopping.step_nan():
                            print("  训练中止（连续NaN过多）。", flush=True)
                            return os.path.join(self.cfg.save_dir, 'vae_best.pt')
                        self.optimizer.zero_grad()
                        continue

                    self.early_stopping.reset_nan()
                    self.scaler.scale(loss / vae_accum).backward()
                    self.global_step += 1

                    if self.global_step % vae_accum == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            list(self.encoder.parameters()) +
                            list(self.decoder.parameters()), 1.0)
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.scaler.get_scale() >= scale_before:
                            self.scheduler.step()
                        self.optimizer.zero_grad()

                    ep_loss  += loss.item()
                    ep_recon += l_recon.item()
                    ep_kl    += l_kl.item()
                    ep_lpips += l_lpips.item()

                    if self.global_step % self.cfg.log_every == 0:
                        self.writer.add_scalar('VAE/loss',  loss.item(),    self.global_step)
                        self.writer.add_scalar('VAE/recon', l_recon.item(), self.global_step)
                        self.writer.add_scalar('VAE/kl',    l_kl.item(),    self.global_step)
                        if self.lpips_fn is not None:
                            self.writer.add_scalar('VAE/lpips', l_lpips.item(), self.global_step)

                    # 显式释放本 batch 大张量，让分配器可回收
                    del loss, l_recon, l_kl, l_lpips

                except torch.cuda.OutOfMemoryError:                        
                    torch.cuda.empty_cache()                                 
                    self.optimizer.zero_grad(set_to_none=True)              
                    skipped += 1                                             
                    print(f"  ！！ OOM: epoch={epoch} batch={batch_idx} 已跳过，显存已清理", flush=True) 
                    continue                                                 

                if batch_idx < 5:
                    gpu_time += time.time() - t_gpu
                if batch_idx == 4 and epoch == 1:
                    avg_data = data_time / 5
                    avg_gpu  = gpu_time  / 5
                    bottleneck = "DataLoader" if avg_data > avg_gpu else "GPU计算"
                    print(f"  [诊断] 前5batch均值: "
                          f"data={avg_data:.3f}s  gpu={avg_gpu:.3f}s  "
                          f"瓶颈={bottleneck}")
                    if avg_data > 0.5:
                        print(f"  [建议] DataLoader耗时过长，"
                              f"考虑增加num_workers或减少数据增强")
                t_data = time.time()   # 重置 data 计时

            n = max(len(self.train_loader) - skipped, 1)
            # epoch 结束后所有 batch 张量已离开作用域
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            val_loss, val_psnr, val_ssim = self.validate()
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:03d}/{self.cfg.vae_epochs:03d} | "
                  f"loss={ep_loss/n:.4f} recon={ep_recon/n:.4f} "
                  f"kl={ep_kl/n:.4f} lpips={ep_lpips/n:.4f} | "
                  f"val={val_loss:.4f} PSNR={val_psnr:.2f}dB SSIM={val_ssim:.4f} | "
                  f"skip={skipped} | {elapsed:.1f}s")
            self.writer.add_scalar('VAE/val_loss', val_loss, epoch)
            self.writer.add_scalar('VAE/val_psnr', val_psnr, epoch)
            self.writer.add_scalar('VAE/val_ssim', val_ssim, epoch)

            if val_loss < self.best_val:
                self.best_val = val_loss
                path = self._save('best')
                print(f"     最优模型已保存: {path}")

            if epoch % self.cfg.save_every == 0:
                self._save(f'epoch{epoch:03d}')

            if epoch % self.cfg.sample_every == 0:
                self._log_samples(epoch)

            if self.early_stopping.step_loss(val_loss, epoch):
                break

        print("  VAE预训练完成！")
        return os.path.join(self.cfg.save_dir, 'vae_best.pt')

    @torch.no_grad()
    def validate(self):
        self.encoder.eval(); self.decoder.eval()
        total_loss = total_psnr = total_ssim = 0.0
        cnt = 0
        for _, hr in self.val_loader:
            try:                                                             
                hr = hr.to(self.device)
                mean, logvar = self.encoder(hr)
                logvar = logvar.clamp(-10, 10)
                std = torch.exp(0.5 * logvar)
                del std
                z = mean
                recon = self.decoder(z)
                del z
                l_recon = F.mse_loss(recon, hr)
                mean_f   = mean.float()
                del mean
                logvar_f = logvar.float()
                del logvar
                l_kl = -0.5 * torch.mean(1 + logvar_f - mean_f.pow(2) - logvar_f.exp())
                del mean_f, logvar_f
                loss = l_recon + self.cfg.vae_kl_weight * l_kl
                del l_recon, l_kl
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    del loss
                    total_psnr += compute_psnr(recon, hr)
                    total_ssim += compute_ssim(recon, hr)
                    cnt += 1
                    del recon, hr
                else:
                    del loss, recon, hr  
            except torch.cuda.OutOfMemoryError:                              
                torch.cuda.empty_cache()                                      
                print("  ！！ validate OOM，跳过此batch，显存已清理", flush=True)  
                continue                                                     
        n = max(cnt, 1)
        return total_loss / n, total_psnr / n, total_ssim / n

    @torch.no_grad()
    def _log_samples(self, epoch):
        self.encoder.eval(); self.decoder.eval()
        try:
            _, hr = next(iter(self.val_loader))
            hr = hr[:4].to(self.device)
            mean, logvar = self.encoder(hr)
            logvar = logvar.clamp(-10, 10)
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(mean)
            del mean, logvar, std
            recon = self.decoder(z)
            del z
            hr_cpu    = hr.cpu()
            recon_cpu = recon.cpu()
            del hr, recon 
            grid = make_grid( torch.cat([hr_cpu, recon_cpu]).clamp(-1, 1) * 0.5 + 0.5, nrow=4)
            del hr_cpu, recon_cpu
            self.writer.add_image('VAE/recon_vs_gt', grid, epoch)
            out_path = os.path.join(self.sample_dir, f'recon_epoch{epoch:03d}.png')
            save_image(grid, out_path)
            print(f"    VAE样本已保存: {out_path}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f" VAE _log_samples OOM，跳过样本保存 (epoch={epoch})", flush=True)


#STAGE 2: 扩散模型训练

class DiffusionTrainer:
    """
    Stage 2: 扩散 UNet 训练。
    • 损失使用 Min-SNR-γ 加权
    • 样本图保存到 samples_sr3/ 
    """
    def __init__(self, cfg, train_loader, val_loader, writer):
        self.cfg          = cfg
        self.device       = cfg.device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.writer       = writer
        self.accum_steps = getattr(cfg, "diff_accum_steps", 4)

        # snr_gamma: 从 cfg 读取
        snr_gamma = getattr(cfg, 'snr_gamma', 5.0)

        self.model = LatentSR3(
            lr_size            = cfg.lr_size,
            hr_size            = cfg.hr_size,
            latent_ch          = cfg.latent_ch,
            vae_base_ch        = cfg.vae_base_ch,
            vae_ch_mult        = getattr(cfg, 'vae_ch_mult',  (1, 2, 4)),
            unet_ch_mult       = getattr(cfg, 'unet_ch_mult', (1, 2, 4, 8)),     
            vae_num_res_blocks = getattr(cfg, 'vae_num_res_blocks', 2),
            unet_base_ch       = cfg.unet_base_ch,
            num_res_blocks     = cfg.num_res_blocks,
            num_timesteps      = cfg.num_timesteps,
            snr_gamma          = snr_gamma,
            lr_enc_ch          = getattr(cfg, 'lr_enc_ch', 64), 
            lr_enc_base_ch     = getattr(cfg, 'lr_enc_base_ch', 32),
            gradient_checkpointing     = getattr(cfg, 'gradient_checkpointing', False),
            vae_gradient_checkpointing = getattr(cfg, 'vae_gradient_checkpointing', False),
        ).to(self.device)

        if cfg.vae_ckpt and os.path.exists(cfg.vae_ckpt):
            self._load_vae(cfg.vae_ckpt)

        self._freeze_vae()
        self.ema = EMA(
            self.model.unet,
            decay           = cfg.ema_decay,
            lazy_ema        = getattr(cfg, 'lazy_ema', False),
            lazy_ema_warmup = getattr(cfg, 'lazy_ema_warmup', 2000),
        )

        self.optimizer = AdamW(
            self.model.unet.parameters(),
            lr=cfg.diff_lr, weight_decay=1e-4, betas=(0.9, 0.999)
        )
        total_steps = cfg.diff_epochs * len(train_loader)
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=cfg.warmup_steps)
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - cfg.warmup_steps, eta_min=1e-6)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[cfg.warmup_steps])

        self.scaler      = torch.amp.GradScaler(enabled=(cfg.fp16 and torch.cuda.is_available()))
        self.global_step = 0
        self.best_val    = float('inf')
        self.start_epoch = 1
        self.early_stopping = EarlyStopping(
            patience  = cfg.early_stop_patience,
            min_delta = cfg.early_stop_min_delta,
            nan_limit = cfg.nan_skip_limit,
        )
        #分阶段样本目录
        self.sample_dir = getattr(cfg, 'save_samples_sr3_dir', './samples_sr3')
        os.makedirs(cfg.save_dir,    exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        if cfg.resume:
            self._resume(cfg.resume)

    def _load_vae(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.encoder.load_state_dict(ckpt['encoder'])
        self.model.decoder.load_state_dict(ckpt['decoder'])
        print(f"  已加载预训练VAE: {path}")

    def _freeze_vae(self):
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        for p in self.model.decoder.parameters():
            p.requires_grad_(False)

        total = sum(p.numel() for p in self.model.unet.parameters()
                    if p.requires_grad) / 1e6
        print(f"  VAE+LREncoder 已冻结，只训练UNet ({total:.2f}M 参数)")

    def _resume(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.ema.load_state_dict(ckpt['ema'])
        self.global_step = ckpt['global_step']
        self.start_epoch = ckpt['epoch'] + 1
        if 'early_stop' in ckpt:
            self.early_stopping.load_state_dict(ckpt['early_stop'])
        print(f"  从 {path} 恢复训练 (epoch={ckpt['epoch']})")

    def _save(self, epoch, tag='latest'):
        path = os.path.join(self.cfg.save_dir, f'sr3_{tag}.pt')
        torch.save({
            'epoch': epoch, 'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'early_stop': self.early_stopping.state_dict(),
        }, path)
        return path

    def _check_model_health(self) -> bool:
        for name, param in self.model.unet.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ✗ 模型参数损坏：{name}")
                return False
        return True

    def train(self):
        print("\n" + "═" * 55)
        print("  STAGE 2: 扩散模型训练（UNet）")
        print(f"  epochs={self.cfg.diff_epochs}  lr={self.cfg.diff_lr}"
              f"  warmup={self.cfg.warmup_steps}"
              f"  snr_gamma={getattr(self.cfg,'snr_gamma',5.0)}")
        print("═" * 55)

        for epoch in range(self.start_epoch, self.cfg.diff_epochs + 1):
            self.model.train()
            self.model.encoder.eval()
            self.model.decoder.eval()
            self.model.lr_encoder.eval()

            ep_loss = 0.0
            t0 = time.time()
            skipped   = 0
            stop_flag = False

            self.optimizer.zero_grad()
            for lr_img, hr_img in self.train_loader:
                try:
                    lr_img = lr_img.to(self.device)
                    hr_img = hr_img.to(self.device)

                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                    with torch.amp.autocast(device_type, enabled=self.cfg.fp16):
                        loss = self.model(hr_img, lr_img)

                    del hr_img, lr_img

                    if not torch.isfinite(loss):
                        skipped += 1
                        del loss 
                        if self.early_stopping.step_nan():
                            self.optimizer.zero_grad(set_to_none=True)
                            stop_flag = True; break
                        self.optimizer.zero_grad()
                        continue

                    self.early_stopping.reset_nan()
                    loss = loss / self.accum_steps
                    self.scaler.scale(loss).backward()

                    if (self.global_step + 1) % self.accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_ok = all(
                            torch.isfinite(p.grad).all()
                            for p in self.model.unet.parameters()
                            if p.grad is not None
                        )
                        if not grad_ok:
                            self.optimizer.zero_grad()
                            self.scaler.update()
                            skipped += 1
                            self.global_step += 1
                            del loss
                            continue
                        nn.utils.clip_grad_norm_(self.model.unet.parameters(), self.cfg.grad_clip)
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.scaler.get_scale() >= scale_before:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.ema.update(self.model.unet)

                    loss_val = loss.item() * self.accum_steps
                    ep_loss += loss_val
                    self.global_step += 1
                    del loss

                    if self.global_step % self.cfg.log_every == 0:
                        lr_now = self.optimizer.param_groups[0]['lr']
                        self.writer.add_scalar('Diff/loss', loss_val, self.global_step)
                        self.writer.add_scalar('Diff/lr',   lr_now,   self.global_step)
                        print(f"    step={self.global_step:06d} "
                              f"loss={loss_val:.5f} lr={lr_now:.2e}", flush=True)

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    print(f"  OOM: batch 已跳过，显存已清理", flush=True)
                    continue

            if stop_flag:
                print("  训练中止（连续NaN过多）。"); break

            # epoch 结束，batch 张量已释放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not self._check_model_health():
                best_path = os.path.join(self.cfg.save_dir, 'sr3_best.pt')
                if os.path.exists(best_path):
                    self._resume(best_path)
                    continue
                else:
                    print("  无可用最佳checkpoint，训练中止。"); break

            n = max(len(self.train_loader) - skipped, 1)
            val_loss, val_psnr, val_ssim = self.validate()
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:03d}/{self.cfg.diff_epochs:03d} | "
                  f"train={ep_loss/n:.5f} val={val_loss:.5f} "
                  f"PSNR={val_psnr:.2f}dB SSIM={val_ssim:.4f} | "
                  f"skip={skipped} | {elapsed:.1f}s")
            self.writer.add_scalars('Diff/epoch_loss',
                                    {'train': ep_loss/n, 'val': val_loss}, epoch)
            self.writer.add_scalar('Diff/val_psnr', val_psnr, epoch)
            self.writer.add_scalar('Diff/val_ssim', val_ssim, epoch)

            self._save(epoch, 'latest')
            if val_loss < self.best_val:
                self.best_val = val_loss
                self._save(epoch, 'best')
                print(f"     最优模型已保存 (val={val_loss:.5f} "
                      f"PSNR={val_psnr:.2f}dB SSIM={val_ssim:.4f})")

            if epoch % self.cfg.save_every == 0:
                self._save(epoch, f'epoch{epoch:03d}')

            if epoch % self.cfg.sample_every == 0:
                self._log_samples(epoch)

            if self.early_stopping.step_loss(val_loss, epoch):
                break

        print("  扩散模型训练完成！")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = total_psnr = total_ssim = 0.0
        cnt = 0

        T_eval = self.model.T // 4  

        for lr_img, hr_img in self.val_loader:
            try:
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                B = hr_img.size(0)
                t_eval = torch.full((B,), T_eval, dtype=torch.long, device=self.device)
                z0  = self.model.encode_mean(hr_img)
                noise = torch.randn_like(z0)
                z_t   = self.model.scheduler.q_sample(z0, t_eval, noise)
                del z0
                lr_cond = self.model.prepare_lr_cond(lr_img)
                del lr_img
                pred_noise = self.model.unet(z_t, t_eval, lr_cond)
                del lr_cond
                loss_per = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1,2,3])
                loss = loss_per.mean()
                del loss_per

                if not torch.isfinite(loss):
                    del noise, z_t, pred_noise, loss, t_eval, hr_img
                    continue

                ac = self.model.scheduler._get('alphas_cumprod', t_eval, z_t.shape)    # (B,1,1,1)，每个样本对应正确的 ac
                del t_eval
                x0_pred = (z_t - (1 - ac).sqrt() * pred_noise) / ac.sqrt()
                del z_t, pred_noise, noise, ac
                x0_pred = x0_pred.clamp(-1, 1)
                sr_pred = self.model.decode(x0_pred)
                del x0_pred

                total_loss += loss.item()
                del loss
                total_psnr += compute_psnr(sr_pred, hr_img)
                total_ssim += compute_ssim(sr_pred, hr_img)
                cnt += 1
                del sr_pred, hr_img
            except torch.cuda.OutOfMemoryError:                                              
                torch.cuda.empty_cache()                                                      
                print(" Diff validate OOM，跳过此batch，显存已清理", flush=True)           
                continue                                                                      

        n = max(cnt, 1)
        return total_loss / n, total_psnr / n, total_ssim / n

    @torch.no_grad()
    def _log_samples(self, epoch):
        self.model.eval()
        orig_sd = {k: v.clone() for k, v in self.model.unet.state_dict().items()}
        try:
            lr_imgs, hr_imgs = next(iter(self.val_loader))
            lr_imgs = lr_imgs[:2].to(self.device)
            hr_imgs = hr_imgs[:2].to(self.device)

            self.ema.apply(self.model.unet)
            sr = self.model.sample_ddim(lr_imgs, num_steps=200)
            self.ema.restore(orig_sd, self.model.unet)

            if not torch.isfinite(sr).all():
                print(f"   样本含NaN/Inf，跳过保存 (epoch={epoch})", flush=True)
                del sr, lr_imgs, hr_imgs
                self.model.train()
                self.model.encoder.eval()
                self.model.decoder.eval()
                self.model.lr_encoder.eval()
                return

            # 计算样本 PSNR/SSIM 并打印
            sample_psnr = compute_psnr(sr, hr_imgs)
            sample_ssim = compute_ssim(sr, hr_imgs)
            self.writer.add_scalar('SR/sample_psnr', sample_psnr, epoch)
            self.writer.add_scalar('SR/sample_ssim', sample_ssim, epoch)

            lr_up = F.interpolate(lr_imgs, size=(self.cfg.hr_size, self.cfg.hr_size),
                                mode='bicubic', align_corners=False)
            B = lr_imgs.size(0)
            interleaved = torch.stack(
                [lr_up, sr, hr_imgs], dim=1          # (B, 3, C, H, W)
            ).reshape(-1, *sr.shape[1:])             # (B*3, C, H, W)
            del lr_up, sr, hr_imgs

            interleaved_cpu = interleaved.cpu()
            del interleaved 
            grid = make_grid(
                interleaved_cpu.clamp(-1, 1) * 0.5 + 0.5,
                nrow=3,         # 每行固定：bicubic | SR | GT
                padding=4,
                pad_value=0.5,  
            )
            del interleaved_cpu
            self.writer.add_image('SR/bicubic_vs_sr_vs_gt', grid, epoch)
            out_path = os.path.join(self.sample_dir, f'sample_epoch{epoch:03d}.png')
            save_image(grid, out_path)
            del grid
            print(f"    SR样本已保存: {out_path}  " f"PSNR={sample_psnr:.2f}dB  SSIM={sample_ssim:.4f}")
            self.model.train()
            self.model.encoder.eval()
            self.model.decoder.eval()
            self.model.lr_encoder.eval()
        except torch.cuda.OutOfMemoryError:
            self.ema.restore(orig_sd, self.model.unet)  # 确保权重一定被还原
            torch.cuda.empty_cache()
            print(f"   _log_samples OOM，跳过样本保存 (epoch={epoch})", flush=True)
            self.model.train()
            self.model.encoder.eval()
            self.model.decoder.eval()
            self.model.lr_encoder.eval()


#推理

@torch.no_grad()
def run_inference(cfg):
    assert cfg.ckpt,   "--ckpt 必须指定模型权重路径"
    assert cfg.lr_img, "--lr_img 必须指定输入LR图片路径"

    device = cfg.device
    model = LatentSR3(
        lr_size            = cfg.lr_size,
        hr_size            = cfg.hr_size,
        latent_ch          = cfg.latent_ch,
        vae_base_ch        = cfg.vae_base_ch,
        vae_ch_mult        = getattr(cfg, 'vae_ch_mult',  (1, 2, 4)),
        unet_ch_mult       = getattr(cfg, 'unet_ch_mult', (1, 2, 4, 8)),     
        vae_num_res_blocks = getattr(cfg, 'vae_num_res_blocks', 2),
        unet_base_ch       = cfg.unet_base_ch,
        num_res_blocks     = cfg.num_res_blocks,
        num_timesteps      = cfg.num_timesteps,
        snr_gamma          = cfg.snr_gamma,
        lr_enc_ch          = getattr(cfg, 'lr_enc_ch', 64),  
        lr_enc_base_ch     = getattr(cfg, 'lr_enc_base_ch', 32),
        gradient_checkpointing     = getattr(cfg, 'gradient_checkpointing', False),
        vae_gradient_checkpointing = getattr(cfg, 'vae_gradient_checkpointing', False),
    ).to(device)

    ckpt = torch.load(cfg.ckpt, map_location=device)
    sd   = ckpt.get('model', ckpt)
    model.load_state_dict(sd)
    model.eval()
    print(f"  已加载模型: {cfg.ckpt}")

    tf = transforms.Compose([
        transforms.Resize((cfg.lr_size, cfg.lr_size),
                          interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    lr = tf(Image.open(cfg.lr_img).convert('RGB')).unsqueeze(0).to(device)

    print(f"  DDIM采样 ({cfg.ddim_steps} 步)...")
    t0 = time.time()
    sr = model.sample_ddim(lr, num_steps=cfg.ddim_steps)
    print(f"  推理耗时: {time.time()-t0:.2f}s")

    sr_img = (sr.squeeze(0).cpu().clamp(-1, 1) + 1) / 2
    save_image(sr_img, cfg.out)
    print(f"  结果已保存: {cfg.out}")



def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    cfg = get_config()

    # 设置 CUDA 分配器碎片控制
    max_mb = getattr(cfg, 'max_alloc_mb', 0)
    if max_mb > 0:
        import os as _os
        _os.environ.setdefault(
            'PYTORCH_CUDA_ALLOC_CONF',
            f'max_split_size_mb:{max_mb}'
        )
        print(f"  [显存] PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:{max_mb}")

    set_seed(cfg.seed)

    print("\n" + "═" * 60)
    print("  Latent SR3 训练框架（v3 — DIV2K + Flickr2K）")
    print(f"  device={cfg.device}  fp16={cfg.fp16}")
    print(f"  显存优化: grad_ckpt={getattr(cfg,'gradient_checkpointing',False)}"
          f"  vae_grad_ckpt={getattr(cfg,'vae_gradient_checkpointing',False)}"
          f"  lazy_ema={getattr(cfg,'lazy_ema',False)}"
          f"  clear_cache={getattr(cfg,'clear_cache_every',100)}"
          f"  pin_memory={getattr(cfg,'pin_memory',True)}"
          f"  max_alloc_mb={getattr(cfg,'max_alloc_mb',0)}")
    print(f"  梯度累积: VAE={getattr(cfg,'vae_accum_steps',1)}步  Diff={getattr(cfg,'diff_accum_steps',4)}步"
          f"  cache_mode={getattr(cfg,'cache_mode','none')}")
    print(f"  train={cfg.train_dir}")
    if cfg.extra_train_dirs:
        print(f"  extra ={cfg.extra_train_dirs}")
    print(f"  valid ={cfg.valid_dir}")
    print(f"  LR={cfg.lr_size}×{cfg.lr_size} -> HR={cfg.hr_size}×{cfg.hr_size}")
    print(f"  增强: hflip={cfg.aug_hflip} vflip={cfg.aug_vflip} "
          f"rotate={cfg.aug_rotate90} crop={cfg.aug_random_crop}")
    print(f"        color={cfg.aug_color_jitter} blur={cfg.aug_blur} "
          f"noise={cfg.aug_noise} mixup={cfg.aug_mixup}")
    print(f"  VAE : epochs={cfg.vae_epochs} batch={cfg.vae_batch} "
          f"lr={cfg.vae_lr} kl={cfg.vae_kl_weight} lpips={cfg.vae_lpips_weight}"
          f"  patch={getattr(cfg,'vae_patch_size',0) or cfg.hr_size}px")
    print(f"  Diff: epochs={cfg.diff_epochs} batch={cfg.diff_batch} "
          f"lr={cfg.diff_lr} snr_gamma={cfg.snr_gamma}")
    print(f"  样本目录: VAE->{cfg.save_samples_vae_dir}  "
          f"SR->{cfg.save_samples_sr3_dir}")
    print("═" * 60)

    if cfg.infer:
        run_inference(cfg)
        return

    # 提前创建两个样本目录
    os.makedirs(cfg.save_samples_vae_dir, exist_ok=True)
    os.makedirs(cfg.save_samples_sr3_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=cfg.log_dir)

    # 按 stage 按需构建数据集，避免重复预载
    # stage=vae  : 只构建 VAE 专用数据集（可能是小 patch），不构建 512px 全集
    # stage=diff : 只构建 512px 全集
    # stage=both : 先构建 VAE 专用集，VAE 训完后再构建 Diff 全集（顺序释放内存）
    train_ds = val_ds = None   # 按需延迟初始化

    vae_ckpt_path = cfg.vae_ckpt
    if cfg.stage in ('vae', 'both'):
        #VAE 专用小分辨率数据集
        vae_patch = getattr(cfg, 'vae_patch_size', 0)
        if vae_patch > 0:
            vae_cfg = type('_', (), {})()
            vae_cfg.__dict__.update(vars(cfg))
            vae_cfg.hr_size    = vae_patch
            vae_cfg.lr_size    = max(vae_patch // 8, 8)
            vae_cfg.crop_size  = vae_patch
            vae_cfg.cache_mode = getattr(cfg, 'cache_mode', 'none')
            vae_train_ds, vae_val_ds = build_datasets(vae_cfg)
            print(f"  [VAE] patch_size={vae_patch}，独立数据集（不预载 512px 全集）")
        else:
            # 无 patch 缩放，直接用全集
            vae_train_ds, vae_val_ds = build_datasets(cfg)
            train_ds, val_ds = vae_train_ds, vae_val_ds  # both 模式可复用

        vae_loader = DataLoader(
            vae_train_ds, batch_size=cfg.vae_batch,
            shuffle=True, num_workers=cfg.num_workers,
            pin_memory=getattr(cfg, 'pin_memory', True))
        vae_val_loader = DataLoader(
            vae_val_ds, batch_size=cfg.vae_batch,
            shuffle=False, num_workers=cfg.num_workers, pin_memory=False)
        trainer = VAETrainer(cfg, vae_loader, vae_val_loader, writer)
        vae_ckpt_path = trainer.train()
        cfg.vae_ckpt  = vae_ckpt_path

        # VAE 训完后释放专用数据集缓存，为 Diff 阶段腾出内存
        if cfg.stage == 'both' and vae_patch > 0:
            del vae_train_ds, vae_val_ds, vae_loader, vae_val_loader
            train_ds = val_ds = None   # 触发 Diff 阶段重新构建

    if cfg.stage in ('diff', 'both'):
        # 如果没有可复用的全集数据集，重新构建
        if train_ds is None:
            train_ds, val_ds = build_datasets(cfg)

        diff_loader = DataLoader(
            train_ds, batch_size=cfg.diff_batch,
            shuffle=True, num_workers=cfg.num_workers,
            pin_memory=getattr(cfg, 'pin_memory', True))
        diff_val_loader = DataLoader(
            val_ds, batch_size=cfg.diff_batch,
            shuffle=False, num_workers=cfg.num_workers, pin_memory=False)
        trainer = DiffusionTrainer(cfg, diff_loader, diff_val_loader, writer)
        trainer.train()

    writer.close()
    print("\n训练全部完成！")


if __name__ == '__main__':
    main()