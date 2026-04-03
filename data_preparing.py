"""
从 HR / X2 / X3 / X4 任意一个来源，生成 X8 下采样数据集，并输出到指定文件夹。

支持的降采样方法：
  - bicubic  (默认)
  - bilinear
  - lanczos
  - nearest

用法示例：
  # 从 HR 图生成 X8
  python data_preparing.py --source_dir /data/Flickr2K/Flickr2K_HR \
      --source_type HR --output_dir /data/Flickr2K/Flickr2K_LR_bicubic/X8

  # 从 X4 图生成 X8（再下采样 ×2）
  python data_preparing.py --source_dir /data/Flickr2K/Flickr2K_LR_bicubic/X4 \
      --source_type X4 --output_dir /data/Flickr2K/Flickr2K_LR_bicubic/X8

  # 指定插值方法
  python data_preparing.py --source_dir /data/Flickr2K/Flickr2K_HR \
      --source_type HR --output_dir ./X8_output --interp lanczos

  # 多线程加速
  python data_preparing.py --source_dir /data/Flickr2K/Flickr2K_HR \
      --source_type HR --output_dir ./X8_output --num_workers 8
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────
# 支持的图片后缀
# ──────────────────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 来源类型 → 目标缩放倍数（相对于 HR，最终得到 X8）
# 比如 X4 来源只需再下采样 ×2；X3 来源再下采样 ×(8/3)；X2 来源再下采样 ×4
SOURCE_TO_SCALE = {
    "HR": 8,
    "X2": 4,
    "X3": 8 / 3,   # ≈ 2.667
    "X4": 2,
}

# Pillow 插值方法映射
INTERP_MAP = {
    "bicubic":  Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "lanczos":  Image.LANCZOS,
    "nearest":  Image.NEAREST,
}


def collect_images(source_dir: Path) -> list[Path]:
    """递归收集目录下所有图片文件。"""
    files = sorted(
        p for p in source_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return files


def downscale_image(
    src_path: Path,
    dst_path: Path,
    scale: float,
    interp: int,
    output_format: str | None,
) -> None:
    """
    对单张图片做 1/scale 下采样，保存到 dst_path。

    Parameters
    ----------
    src_path    : 源图路径
    dst_path    : 目标图路径
    scale       : 缩放倍数（> 1 表示缩小）
    interp      : Pillow 插值常量
    output_format: 强制输出格式（如 "PNG"）；None 则与源文件相同
    """
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    new_w = max(1, round(w / scale))
    new_h = max(1, round(h / scale))
    img_resized = img.resize((new_w, new_h), resample=interp)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format or src_path.suffix.lstrip(".").upper()
    # JPEG → JPEG；PNG → PNG；其他统一用 PNG
    if fmt in ("JPG", "JPEG"):
        img_resized.save(dst_path, format="JPEG", quality=95)
    else:
        img_resized.save(dst_path, format="PNG")


def build_dst_path(
    src_path: Path,
    source_dir: Path,
    output_dir: Path,
    output_ext: str | None,
) -> Path:
    """根据源路径计算目标路径（保持子目录结构）。"""
    rel = src_path.relative_to(source_dir)
    if output_ext:
        rel = rel.with_suffix(output_ext)
    return output_dir / rel


def process_one(args_tuple):
    """线程池工作函数（解包参数）。"""
    src, dst, scale, interp, fmt = args_tuple
    try:
        downscale_image(src, dst, scale, interp, fmt)
        return None
    except Exception as exc:
        return f"[ERROR] {src}: {exc}"


def main():
    parser = argparse.ArgumentParser(
        description="从 Flickr2K HR/X2/X3/X4 生成 X8 低分辨率数据集"
    )
    parser.add_argument(
        "--source_dir", type=str, required=True,
        help="源图像目录（HR / X2 / X3 / X4 任意一个）"
    )
    parser.add_argument(
        "--source_type", type=str, required=True,
        choices=["HR", "X2", "X3", "X4"],
        help="源目录的类型：HR、X2、X3 或 X4"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="X8 输出目录"
    )
    parser.add_argument(
        "--interp", type=str, default="bicubic",
        choices=list(INTERP_MAP.keys()),
        help="插值方法（默认：bicubic）"
    )
    parser.add_argument(
        "--output_ext", type=str, default=None,
        help="强制输出扩展名，如 .png（默认：与源文件相同）"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="并行线程数（默认：4）"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="试运行：只打印将处理的文件，不实际写入"
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    scale      = SOURCE_TO_SCALE[args.source_type]
    interp     = INTERP_MAP[args.interp]
    output_ext = args.output_ext  # e.g. ".png" or None

    # ── 检查源目录 ──────────────────────────────
    if not source_dir.is_dir():
        print(f"[ERROR] 源目录不存在：{source_dir}")
        sys.exit(1)

    print(f"📂 源目录   : {source_dir}")
    print(f"📦 源类型   : {args.source_type}  →  下采样倍数 ×{scale:.4g}")
    print(f"🎯 输出目录 : {output_dir}")
    print(f"🔧 插值方法 : {args.interp}")
    print(f"⚙️  线程数   : {args.num_workers}")
    print()

    # ── 收集图片 ────────────────────────────────
    images = collect_images(source_dir)
    if not images:
        print("[WARN] 未找到任何图片，请检查 --source_dir 路径与文件格式。")
        sys.exit(0)
    print(f"✅ 共找到 {len(images)} 张图片")

    if args.dry_run:
        print("\n[DRY RUN] 前 10 个将处理的文件：")
        for p in images[:10]:
            dst = build_dst_path(p, source_dir, output_dir, output_ext)
            print(f"  {p}  →  {dst}")
        print("（dry_run 模式，不执行实际写入）")
        return

    # ── 创建输出目录 ────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 构建任务列表 ────────────────────────────
    tasks = [
        (
            img,
            build_dst_path(img, source_dir, output_dir, output_ext),
            scale,
            interp,
            output_ext.lstrip(".").upper() if output_ext else None,
        )
        for img in images
    ]

    # ── 多线程处理 ──────────────────────────────
    errors = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), desc="生成 X8", unit="img") as pbar:
            for future in as_completed(futures):
                err = future.result()
                if err:
                    errors.append(err)
                pbar.update(1)

    # ── 结果汇报 ────────────────────────────────
    print()
    success = len(tasks) - len(errors)
    print(f"✅ 成功：{success} 张  |  ❌ 失败：{len(errors)} 张")
    if errors:
        print("\n失败列表：")
        for e in errors:
            print(f"  {e}")

    print(f"\n🎉 X8 数据集已输出至：{output_dir}")


if __name__ == "__main__":
    main()