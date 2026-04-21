import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List
import os


#PSNR / SSIM

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 2.0) -> float:
    """
    计算 PSNR
    """
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])                   # (B,)
    psnr = 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-10))
    return psnr.mean().item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11,
                 data_range: float = 2.0) -> float:
    """
    计算 SSIM
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    B, C, H, W = pred.shape
    # 构建高斯核
    coords = torch.arange(window_size, dtype=torch.float32,
                          device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = g.outer(g).view(1, 1, window_size, window_size)
    kernel = kernel.expand(C, 1, window_size, window_size)

    pred   = pred.float()
    target = target.float()
    kernel = kernel.contiguous()

    pad = window_size // 2

    def _filt(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1 = _filt(pred)
    mu2 = _filt(target)
    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filt(pred   * pred)    - mu1_sq
    sigma2_sq = _filt(target * target)  - mu2_sq
    sigma12   = _filt(pred   * target)  - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


class LPIPSLoss(nn.Module):
    """
    轻量 LPIPS
    """

    def __init__(self):
        super().__init__()
        # 使用 torchvision VGG16 前几层作为特征提取器
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        # 取 relu1_2, relu2_2, relu3_3 三层
        self.slice1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9]).eval()
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]).eval()

        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet 归一化参数（输入先从 [-1,1] 转到 [0,1]，再 ImageNet-norm）
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] -> ImageNet-normalized."""
        x = (x + 1.0) / 2.0          #[0,1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        p = self._normalize(pred)
        t = self._normalize(target)

        if p.shape[-1] > 256:
            p = F.interpolate(p, size=(224, 224), mode='bilinear', align_corners=False)
            t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)

        feat_p, feat_t = p, t
        loss = 0.0
        for sl in (self.slice1, self.slice2, self.slice3):
            feat_p = sl(feat_p)
            feat_t = sl(feat_t)
            p_norm = F.normalize(feat_p, dim=1)
            t_norm = F.normalize(feat_t, dim=1)
            loss = loss + F.mse_loss(p_norm, t_norm)

        return loss


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half = self.dim // 2
        embeddings = math.log(10000) / max(half - 1, 1)
        embeddings = torch.exp(torch.arange(half, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 cond_ch: int = 0, groups: int = 8, dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)
        )
        self.cond_proj = nn.Conv2d(cond_ch, in_ch, 1) if cond_ch > 0 else None

        self.norm1   = nn.GroupNorm(groups, in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2   = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def _forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                 cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        #条件注入
        if cond is not None and self.cond_proj is not None:
            assert cond.shape[-2:] == x.shape[-2:], \
                f"cond shape {cond.shape} != x shape {x.shape}, should be pre-aligned"
            x = x + self.cond_proj(cond)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # time scale + shift 
        t = self.time_mlp(time_emb)[:, :, None, None]
        scale, shift = t.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.res_conv(x)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            if cond is None:
                cond_in = torch.zeros(1, device=x.device, requires_grad=False)
                def _ckpt_fn(x_, te_, _dummy):
                    return self._forward(x_, te_, None)
                return torch_checkpoint.checkpoint(
                    _ckpt_fn, x, time_emb, cond_in,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                return torch_checkpoint.checkpoint(
                    self._forward, x, time_emb, cond,
                    use_reentrant=False, preserve_rng_state=False)
        return self._forward(x, time_emb, cond)


class AttentionBlock(nn.Module):

    def __init__(self, ch: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        assert ch % num_heads == 0, f"ch={ch} 必须整除 num_heads={num_heads}"
        self.heads   = num_heads
        self.head_dim = ch // num_heads
        self.norm    = nn.GroupNorm(groups, ch)
        self.qkv     = nn.Conv2d(ch, ch * 3, 1)
        self.proj    = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h)                                       # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3, self.heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)                                 # 各 (B, heads, head_dim, HW)
        # F.sdpa 期望 (B, heads, seq, head_dim)
        q = q.transpose(-2, -1)    # (B, heads, HW, head_dim)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(-2, -1).reshape(B, C, H, W)         # (B, C, H, W)
        return x + self.proj(out)

class CrossAttentionCondBlock(nn.Module):
    """
    将 LR latent 条件以 Cross-Attention 方式注入到特征图。
    """
    def __init__(self, feat_ch: int, cond_ch: int,
                 num_heads: int = 4, groups: int = 8):
        super().__init__()
        assert feat_ch % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = feat_ch // num_heads

        self.norm_feat = nn.GroupNorm(groups, feat_ch)
        self.norm_cond = nn.GroupNorm(min(groups, cond_ch), cond_ch)

        self.q_proj = nn.Conv2d(feat_ch, feat_ch, 1)
        self.k_proj = nn.Conv2d(cond_ch, feat_ch, 1)
        self.v_proj = nn.Conv2d(cond_ch, feat_ch, 1)
        self.out_proj = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)   # 零初始化

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        # cond 对齐到 h 的空间尺寸
        if cond.shape[-2:] != h.shape[-2:]:
            cond = F.interpolate(cond, size=h.shape[-2:],
                                 mode='bilinear', align_corners=False)

        q = self.q_proj(self.norm_feat(h))
        k = self.k_proj(self.norm_cond(cond))
        v = self.v_proj(self.norm_cond(cond))

        # reshape 为 (B, heads, HW, head_dim)
        def _split(t):
            return t.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2,-1)

        out = F.scaled_dot_product_attention(_split(q), _split(k), _split(v))
        out = out.transpose(-2,-1).reshape(B, C, H, W)
        return h + self.out_proj(out)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


#  Conditional UNet
class ConditionalUNet(nn.Module):
    """
    UNet denoising backbone，LR 条件注入所有层级每个 ResidualBlock。
    """

    def __init__(
        self,
        in_ch:  int = 4,
        out_ch: int = 4,
        cond_ch: int = 4,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (64, 32, 16),
        time_emb_dim: int = 512,
        dropout: float = 0.1,
        latent_size: int = 32,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.latent_size    = latent_size
        self.use_checkpoint = use_checkpoint
        self.cond_ch     = cond_ch

        #时间嵌入 MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        channels   = [base_ch * m for m in ch_mult]
        num_levels = len(channels)

        #Encoder（下采样路径）
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        curr_res = latent_size
        prev_ch  = base_ch

        for i, ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResidualBlock(
                    prev_ch, ch, time_emb_dim,
                    cond_ch=0, 
                    dropout=dropout,
                    use_checkpoint=use_checkpoint,
                ))
                level_blocks.append(CrossAttentionCondBlock(
                    feat_ch=ch, cond_ch=cond_ch,
                    num_heads=max(1, ch // 64),
                ))
                if curr_res in attn_resolutions:
                    level_blocks.append(AttentionBlock(ch))
                prev_ch = ch
            self.down_blocks.append(level_blocks)
            if i < num_levels - 1:
                self.downsamples.append(Downsample(ch))
                curr_res //= 2
            else:
                self.downsamples.append(nn.Identity())

        #Bottleneck
        mid_ch = channels[-1]
        self.mid_res1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim,
                                      cond_ch=0,
                                      dropout=dropout,
                                      use_checkpoint=use_checkpoint)
        self.mid_cross1 = CrossAttentionCondBlock(
            feat_ch=mid_ch, cond_ch=cond_ch,
            num_heads=max(1, mid_ch // 64),
        )
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_res2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim,
                                      cond_ch=0,
                                      dropout=dropout,
                                      use_checkpoint=use_checkpoint)
        self.mid_cross2 = CrossAttentionCondBlock(
            feat_ch=mid_ch, cond_ch=cond_ch,
            num_heads=max(1, mid_ch // 64),
        )

        #Decoder（上采样路径）
        self.up_blocks  = nn.ModuleList()
        self.upsamples  = nn.ModuleList()

        for i in reversed(range(num_levels)):
            if i < num_levels - 1:
                curr_res *= 2                       
            ch      = channels[i]
            skip_ch = channels[i]
            level_blocks = nn.ModuleList()
            in_channels  = prev_ch + skip_ch
            for j in range(num_res_blocks + 1):
                out_c = channels[i - 1] if (i > 0 and j == num_res_blocks) else ch
                level_blocks.append(ResidualBlock(
                    in_channels, out_c, time_emb_dim,
                    cond_ch=0, dropout=dropout, use_checkpoint=use_checkpoint,
                ))
                level_blocks.append(CrossAttentionCondBlock(
                    feat_ch=out_c, cond_ch=cond_ch,
                    num_heads=max(1, out_c // 64),
                ))
                if curr_res in attn_resolutions:    
                    level_blocks.append(AttentionBlock(out_c))
                in_channels = out_c
                prev_ch     = out_c
            self.up_blocks.append(level_blocks)
            if i > 0:
                self.upsamples.append(Upsample(prev_ch))
            else:
                self.upsamples.append(nn.Identity())

        # 输出头
        _g = min(8, prev_ch)
        while prev_ch % _g != 0:
            _g -= 1
        self.out_norm = nn.GroupNorm(_g, prev_ch)
        self.out_conv = nn.Conv2d(prev_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, lr_latent: torch.Tensor) -> torch.Tensor:
        def _run_block(blk, h, time_emb, cond):
            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(blk, h, time_emb, cond, use_reentrant=False)
            return blk(h, time_emb, cond)
        
        time_emb = self.time_embed(t)
        h = self.init_conv(x)

        #Encoder
        skips = []
        for i, (level_blocks, down) in enumerate(zip(self.down_blocks, self.downsamples)):
            cond = F.interpolate(lr_latent, size=h.shape[-2:],
                    mode='bilinear', align_corners=False)
            for blk in level_blocks:
                if isinstance(blk, ResidualBlock):
                    h = _run_block(blk, h, time_emb, None)      
                elif isinstance(blk, CrossAttentionCondBlock):
                    h = blk(h, cond)                            
                else:
                    h = blk(h)                                  
            skips.append(h)
            h = down(h)

        #Bottleneck
        cond_bot = F.interpolate(lr_latent, size=h.shape[-2:],
                    mode='bilinear', align_corners=False)
        h = _run_block(self.mid_res1, h, time_emb, None)
        h = self.mid_cross1(h, cond_bot)          
        h = self.mid_attn(h)
        h = _run_block(self.mid_res2, h, time_emb, None)
        h = self.mid_cross2(h, cond_bot)         

        #Decoder
        for i, (level_blocks, up) in enumerate(zip(self.up_blocks, self.upsamples)):
            h = torch.cat([h, skips.pop()], dim=1)
            cond = F.interpolate(lr_latent, size=h.shape[-2:],
                    mode='bilinear', align_corners=False)
            for blk in level_blocks:
                if isinstance(blk, ResidualBlock):
                    h = _run_block(blk, h, time_emb, None)  
                elif isinstance(blk, CrossAttentionCondBlock):
                    h = blk(h, cond)                        
                else:
                    h = blk(h)            
            h = up(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


#VAE (Encoder + Decoder)

class _VAEResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        groups_in  = min(8, in_ch)
        while in_ch % groups_in != 0:
            groups_in -= 1
        groups_out = min(8, out_ch)
        while out_ch % groups_out != 0:
            groups_out -= 1

        self.net = nn.Sequential(
            nn.GroupNorm(groups_in, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups_out, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint.checkpoint(
                self._forward, x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward(x)

#LR 专用编码器

class LREncoder(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 64,
                 base_ch: int = 32, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.body = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),   # /2
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, stride=2, padding=1),    # /4
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

        # 原始像素直接 avgpool 到 /4 尺寸，再投影到 out_ch
        # 保留低频结构信息，与主干特征相加
        self.skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),   # /4，与主干对齐
            nn.Conv2d(in_ch, out_ch, 1),             # 通道对齐
        )

    def _forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        return self.body(lr_img) + self.skip(lr_img)

    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and lr_img.requires_grad:
            return torch_checkpoint.checkpoint(
                self._forward, lr_img, use_reentrant=False)
        return self._forward(lr_img)

class VAEEncoder(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base_ch=128,
                 ch_mult=(1,2,4,8), num_res_blocks=2,
                 use_checkpoint=False,
                 attn_at_levels=None):  
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.levels      = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = base_ch

        n_levels = len(ch_mult)
        if attn_at_levels is None:
            attn_at_levels = set(range(max(0, n_levels - 2), n_levels))
        else:
            attn_at_levels = set(attn_at_levels)

        for level_idx, m in enumerate(ch_mult):
            ch = base_ch * m
            blocks = nn.Sequential()
            for i in range(num_res_blocks):
                blocks.append(_VAEResBlock(prev if i == 0 else ch, ch,
                                           use_checkpoint=use_checkpoint))
            if level_idx in attn_at_levels:
                num_heads = max(1, ch // 64)
                groups    = min(8, ch)
                while ch % groups != 0:
                    groups -= 1
                blocks.append(AttentionBlock(ch, num_heads=num_heads, groups=groups))
            self.levels.append(blocks)
            self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev = ch

        groups = min(8, prev)
        while prev % groups != 0:
            groups -= 1
        self.out = nn.Sequential(
            nn.GroupNorm(groups, prev),
            nn.SiLU(),
            nn.Conv2d(prev, latent_ch * 2, 1),
        )

    def forward(self, x):
        h = self.init_conv(x)
        for blocks, down in zip(self.levels, self.downsamples):
            if self.use_checkpoint and h.requires_grad:
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(blocks, h, use_reentrant=False)
            else:
                h = blocks(h)
            h = down(h)
        out = self.out(h)
        mean, logvar = out.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        return mean, logvar

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)


class VAEDecoder(nn.Module):

    def __init__(self, out_ch: int = 3, latent_ch: int = 4,
                base_ch: int = 128, ch_mult=(8, 4, 2, 1),
                num_res_blocks: int = 2,
                use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        ch_list = [base_ch * m for m in ch_mult]

        self.init_conv = nn.Conv2d(latent_ch, ch_list[0], 3, padding=1)

        mid_ch = ch_list[0]
        num_heads = max(1, mid_ch // 64)
        groups = min(8, mid_ch)
        while mid_ch % groups != 0:
            groups -= 1
        self.mid_attn = AttentionBlock(mid_ch, num_heads=num_heads, groups=groups)

        self.upsamples = nn.ModuleList()
        self.levels    = nn.ModuleList()
        prev = ch_list[0]
        for ch in ch_list[1:]:
            # 上采样：nearest + Conv 消除棋盘格伪影
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(prev, ch, 3, padding=1),
            ))
            # ResBlock 序列（含残差连接）
            blocks = nn.Sequential()
            for i in range(num_res_blocks):
                blocks.append(_VAEResBlock(ch, ch, use_checkpoint=use_checkpoint))
            self.levels.append(blocks)
            prev = ch

        # 最后一次上采样 + 输出头
        groups = min(8, prev)
        while prev % groups != 0:
            groups -= 1
        self.out = nn.Sequential(
            nn.GroupNorm(groups, prev),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(prev, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        h = self.init_conv(z)
        h = self.mid_attn(h)
        for up, blocks in zip(self.upsamples, self.levels):
            h = up(h)
            if self.use_checkpoint and h.requires_grad:
                h = checkpoint(blocks, h, use_reentrant=False)
            else:
                h = blocks(h)
        return self.out(h)


#DDPM

class DDPMScheduler(nn.Module):  
    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()        
        self.T = num_timesteps

        def _cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(
                ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(0.0001, 0.9999)

        betas              = _cosine_beta_schedule(num_timesteps)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_var      = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register_buffer('betas',                        betas)
        self.register_buffer('alphas_cumprod',               alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',          alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',          alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',(1 - alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recip_alphas_cumprod',    (1 / alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recipm1_alphas_cumprod',  (1 / alphas_cumprod - 1).sqrt())
        self.register_buffer('posterior_variance',           posterior_var)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_var.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod))
        self.register_buffer('snr',
                             alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-10))

    def _get(self, name: str, t: torch.Tensor, shape) -> torch.Tensor:
        vals = getattr(self, name)
        out  = vals.gather(0, t)
        return out.reshape(len(t), *((1,) * (len(shape) - 1)))

    def snr_weights(self, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        snr_t = self.snr.gather(0, t)
        return torch.minimum(snr_t, torch.full_like(snr_t, gamma)) / snr_t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sa  = self._get('sqrt_alphas_cumprod',            t, x_start.shape)
        s1a = self._get('sqrt_one_minus_alphas_cumprod',  t, x_start.shape)
        return sa * x_start + s1a * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (self._get('sqrt_recip_alphas_cumprod',   t, x_t.shape) * x_t
              - self._get('sqrt_recipm1_alphas_cumprod', t, x_t.shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        mean = (self._get('posterior_mean_coef1', t, x_t.shape) * x_start
              + self._get('posterior_mean_coef2', t, x_t.shape) * x_t)
        log_var = self._get('posterior_log_variance_clipped', t, x_t.shape)
        return mean, log_var

    @torch.no_grad()
    def p_sample(self, model_output, x_t, t):
        x_start = self.predict_start_from_noise(x_t, t, model_output)
        x_start = x_start.clamp(-1, 1)
        mean, log_var = self.q_posterior(x_start, x_t, t)
        noise   = torch.randn_like(x_t)
        nonzero = (t > 0).float().reshape(len(t), *([1] * (x_t.ndim - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise

#  Full Latent SR3 Model（改：encode_mean + Min-SNR-γ）

class LatentSR3(nn.Module):

    def __init__(
        self,
        lr_size:            int   = 64,
        hr_size:            int   = 512,
        latent_ch:          int   = 4,
        vae_base_ch:        int   = 128,
        vae_ch_mult:        Tuple[int, ...] = (1, 2, 4),
        vae_num_res_blocks: int   = 2,
        unet_base_ch:       int   = 128,
        unet_ch_mult:       Tuple[int, ...] = (1, 2, 4),
        num_res_blocks:     int   = 2,
        num_timesteps:      int   = 2000,
        attn_resolutions:   Tuple[int, ...] = (64, 32, 16),
        snr_gamma:          float = 1.0,
        lr_enc_ch:          int   = 64,    # LREncoder 输出通道数
        lr_enc_base_ch:     int   = 32,    # LREncoder 中间通道数
        gradient_checkpointing:     bool = False,
        vae_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        assert hr_size % lr_size == 0, \
            f"hr_size({hr_size}) 必须整除 lr_size({lr_size})"

        self.lr_size   = lr_size
        self.hr_size   = hr_size
        self.latent_ch = latent_ch
        self.snr_gamma = snr_gamma

        # VAE 每个 ch_mult 层级包含一次 stride=2 下采样
        self.vae_factor     = 2 ** len(vae_ch_mult)         
        self.hr_latent_size = hr_size // self.vae_factor     
        self.lr_latent_size = lr_size // self.vae_factor     

        #VAE
        self.encoder = VAEEncoder(
            in_ch=3, latent_ch=latent_ch,
            base_ch=vae_base_ch, ch_mult=vae_ch_mult,
            num_res_blocks=vae_num_res_blocks,
            use_checkpoint=vae_gradient_checkpointing,
        )
        self.decoder = VAEDecoder(
            out_ch=3, latent_ch=latent_ch,
            base_ch=vae_base_ch, ch_mult=tuple(reversed(vae_ch_mult)),
            num_res_blocks=vae_num_res_blocks,
            use_checkpoint=vae_gradient_checkpointing,
        )

        # 下采样
        self.lr_enc_ch = lr_enc_ch
        self.lr_encoder = LREncoder(
            in_ch=3, out_ch=lr_enc_ch,
            base_ch=lr_enc_base_ch,
            use_checkpoint=gradient_checkpointing,
        )
        self.lr_feat_size = lr_size // 4

        #去噪 UNet
        self.unet = ConditionalUNet(
            in_ch=latent_ch,
            out_ch=latent_ch,
            cond_ch=lr_enc_ch,         
            base_ch=unet_base_ch,
            ch_mult=unet_ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            latent_size=self.hr_latent_size,
            use_checkpoint=gradient_checkpointing,
        )

        #DDPM 调度器
        self.scheduler = DDPMScheduler(num_timesteps=num_timesteps)
        self.T = num_timesteps

    #VAE 工具

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return self.encoder.sample(img)

    def encode_mean(self, img: torch.Tensor) -> torch.Tensor:
        mean, _ = self.encoder(img)
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


    def prepare_lr_cond(self, lr_img: torch.Tensor) -> torch.Tensor:
        """
        用 LREncoder 直接从像素空间提取特征，下采样
        """
        return self.lr_encoder(lr_img)

    #训练 forward

    def forward(self, hr_img: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        B      = hr_img.size(0)
        device = hr_img.device

        with torch.no_grad():
            mean, logvar = self.encoder(hr_img)
            logvar = logvar.clamp(-10, 10)
            std    = torch.exp(0.5 * logvar)
            z0 = mean + std * torch.randn_like(std)
            mean_f = mean.float()
            logvar_f = logvar.float()
            kl_loss = -0.5 * (1 + logvar_f - mean_f.pow(2) - logvar_f.exp()).mean()
            del mean, logvar, std, mean_f, logvar_f

        t      = torch.randint(0, self.T, (B,), device=device)
        noise  = torch.randn_like(z0)
        z_t    = self.scheduler.q_sample(z0, t, noise)
        del z0 
        lr_cond = self.prepare_lr_cond(lr_img)          # (B, lr_enc_ch, 16, 16)
        pred_noise = self.unet(z_t, t, lr_cond)
        del z_t, lr_cond

        loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1,2,3])
        del pred_noise, noise
        if self.snr_gamma is not None and not math.isinf(self.snr_gamma):
            weights   = self.scheduler.snr_weights(t, gamma=self.snr_gamma)
            diff_loss = (weights * loss_per_sample).mean()
            del weights
        else:
            diff_loss = loss_per_sample.mean()

        del loss_per_sample, t

        return diff_loss + 1e-4 * kl_loss

    #DDPM 全步推理
    @torch.no_grad()
    def sample(self, lr_img: torch.Tensor, num_inference_steps: Optional[int] = None) -> torch.Tensor:
        B      = lr_img.size(0)
        device = lr_img.device
        steps  = num_inference_steps or self.T

        lr_cond = self.prepare_lr_cond(lr_img)

        z = torch.randn(B, self.latent_ch,
                        self.hr_latent_size, self.hr_latent_size,
                        device=device)

        timesteps = torch.linspace(self.T - 1, 0, steps,
                                   dtype=torch.long, device=device)
        for step in timesteps:
            t_batch    = step.expand(B)
            pred_noise = self.unet(z, t_batch, lr_cond)
            z          = self.scheduler.p_sample(pred_noise, z, t_batch)

        return self.decode(z)

    #DDIM 快速推理

    @torch.no_grad()
    def sample_ddim(self, lr_img: torch.Tensor, num_steps: int = 200, eta: float = 0.0) -> torch.Tensor:
        num_steps = min(num_steps, self.T)

        B      = lr_img.size(0)
        device = lr_img.device
        lr_cond = self.prepare_lr_cond(lr_img)

        z = torch.randn(B, self.latent_ch,
                        self.hr_latent_size, self.hr_latent_size,
                        device=device)

        acp = self.scheduler.alphas_cumprod

        # 均匀选取时间步序列（从 T-1 到 0）
        step_indices = torch.linspace(0, self.T - 1, num_steps + 1,
                                      dtype=torch.long, device=device).flip(0)

        for i in range(num_steps):
            t_cur  = step_indices[i].expand(B)

            ac_cur  = acp[step_indices[i]].view(1, 1, 1, 1)
            ac_prev = acp[step_indices[i + 1]].view(1, 1, 1, 1)

            pred_noise = self.unet(z, t_cur, lr_cond)
            del t_cur

            # DDIM 更新
            x0_pred = (z - (1 - ac_cur).sqrt() * pred_noise) / ac_cur.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = eta * ((1 - ac_prev) / (1 - ac_cur) * (1 - ac_cur / ac_prev)).sqrt()
            del ac_cur
            noise = torch.randn_like(z) if eta > 0 else torch.zeros_like(z)

            dir_coef = (1 - ac_prev - sigma ** 2).clamp(min=0.0).sqrt()
            z = ac_prev.sqrt() * x0_pred + dir_coef * pred_noise + sigma * noise
            del pred_noise, x0_pred, sigma, noise, dir_coef, ac_prev

        return self.decode(z)


class SRDataset(Dataset):
    """配对 SR 数据集，LR 由 HR 双三次下采样生成。"""

    def __init__(self, hr_paths: List[str]):
        self.hr_paths    = hr_paths
        self.hr_transform = transforms.Compose([
            transforms.Resize((512, 512),
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((64, 64),
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        img = Image.open(self.hr_paths[idx]).convert('RGB')
        return self.lr_transform(img), self.hr_transform(img)


def run_sanity_check():
    print("=" * 60)
    print("  Latent SR3")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device : {device}\n")

    model = LatentSR3(
        lr_size=64, hr_size=512, latent_ch=4,
        vae_base_ch=128, vae_num_res_blocks=2,
        unet_base_ch=64,
        unet_ch_mult=(1, 2, 4),
        num_res_blocks=1, num_timesteps=100,
        snr_gamma=5.0,
    ).to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters : {total:.2f}M")

    B      = 2
    lr_img = torch.randn(B, 3, 64, 64).to(device)
    hr_img = torch.randn(B, 3, 512, 512).to(device)

    # 训练 forward
    loss = model(hr_img, lr_img)
    print(f"  Training loss    : {loss.item():.5f}  ✓")

    # DDIM 推理
    model.eval()
    sr_out = model.sample_ddim(lr_img[:1], num_steps=5)
    print(f"  DDIM output shape: {tuple(sr_out.shape)}  ✓")

    # VAE 编解码
    z   = model.encode_mean(hr_img[:1])
    rec = model.decode(z)
    print(f"  VAE latent shape : {tuple(z.shape)} ")
    print(f"  VAE recon  shape : {tuple(rec.shape)} ")

    # PSNR / SSIM
    psnr = compute_psnr(rec, hr_img[:1])
    ssim = compute_ssim(rec, hr_img[:1])
    print(f"  Random PSNR      : {psnr:.2f} dB ")
    print(f"  Random SSIM      : {ssim:.4f}")

    # LPIPS（可选）
    try:
        lpips_fn = LPIPSLoss().to(device)
        pl = lpips_fn(rec, hr_img[:1])
        print(f"  LPIPS loss       : {pl.item():.5f} ")
    except Exception as e:
        print(f"  LPIPS skip       : {e}")

    # Min-SNR 权重验证
    T = model.T
    t_batch = torch.tensor([0, T // 2, T - 1], dtype=torch.long, device=device)
    w = model.scheduler.snr_weights(t_batch, gamma=5.0)
    print(f"  Min-SNR weights (t=0, T/2, T-1): {w.tolist()} ")

    print("\nAll checks passed!")


if __name__ == '__main__':
    run_sanity_check()