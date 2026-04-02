#!/usr/bin/env bash

# 一旦有任何命令失败就退出脚本
set -e

# # 1. 创建并激活 conda 环境
# echo "创建 conda 环境 pl ..."
# conda create -n pl python=3.9 -y

# # 激活 conda 环境（兼容 bash/zsh）
# echo "激活 conda 环境 pl ..."
# eval "$(conda shell.bash hook)"
# conda activate pl
# python=3.12
# 2. 安装 PyTorch、TorchVision、TorchAudio（CUDA 11.8）
echo "安装 torch 及相关包 ..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 MMCV（指定 mmcv 2.2.0 + CUDA11.8 + PyTorch2.4 版本）
echo "安装 mmcv ..."
pip install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html

# 4. 安装 OpenMIM 并通过 MIM 安装 mmengine
echo "安装 openmim 并安装 mmengine ..."
pip install -U openmim
pip install -U setuptools
mim install mmengine

# 5. 安装其他依赖
echo "安装其他依赖包 ftfy、regex、lightning、albumentations、timm、einops、mmsegmentation ..."
pip install fvcore
pip install -U comet-ml
pip install ftfy regex
pip install lightning==2.4.0
conda install -c conda-forge albumentations -y
pip install timm
pip install einops
pip install -U numpy scipy
pip install hydra-core
pip install peft
pip install "mmsegmentation>=1.0.0"

echo "✔ 环境安装完成，请确保已激活 conda 环境 pl，再运行你的代码。"

echo "✔ 会遇到下面这个bug，因为mmsegmentation与mmcv的版本冲突问题"

cat << 'EOF'
Traceback (most recent call last):
  File "/home/jic2/git_code/ChangeDetection/train_pl.py", line 11, in <module>
    from model.seed import DEDD_SwinT
  File "/home/jic2/git_code/ChangeDetection/model/seed.py", line 6, in <module>
    from mmseg.registry import MODELS
  File "/home/jic2/miniconda3/envs/pl/lib/python3.9/site-packages/mmseg/__init__.py", line 61, in <module>
    assert (mmcv_min_version <= mmcv_version < mmcv_max_version), \
AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4.
EOF

echo "✔ 解决方法："

echo "修改为： assert (mmcv_min_version <= mmcv_version <= mmcv_max_version)" 