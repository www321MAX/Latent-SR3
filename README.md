A Latent SR3 model code
usage:
  两阶段完整训练
  python train_sr3.py --stage both --fp16
      --train_dir ./data/train/div2k
      --extra_train_dirs ./data/train/flickr2k
      --valid_dir ./data/valid

  只训练扩散模型（已有VAE权重）
  python train_sr3.py --stage diff --fp16
      --vae_ckpt ./checkpoints/vae_best.pt

  推理
  python train_sr3.py --infer
      --lr_img ./test_lr.png
      --ckpt ./checkpoints/sr3_best.pt
