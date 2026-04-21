# PeftCD: Leveraging Vision Foundation Models with Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/[insert_arxiv_if_available])   -->
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/dyzy41/PeftCD)

Official PyTorch implementation of **PeftCD**, a novel bitemporal change detection network that utilize the VFMs fintuning methods in remote sensing change detection. The paper has been accepted at [IEEE JSTAR](https://ieeexplore.ieee.org/document/11458815)!!

## Paper

**Title:** [PeftCD: Leveraging Vision Foundation Models with Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection](https://ieeexplore.ieee.org/document/11458815)

**Authors:** Sijun Dong; Yuxuan Hu; Libo Wang; Geng Chen; Xiaoliang Meng*  
*School of Remote Sensing and Information Engineering, Wuhan University, Wuhan, China*  

**Abstract:**

To tackle the prevalence of spurious changes, the scarcity of annotations, and the difficulty of cross-domain transfer in multi-temporal and multi-source remote sensing imagery, we propose PeftCD, a change detection framework built upon Vision Foundation Models (VFMs) with Parameter-Efficient Fine-Tuning (PEFT). Specifically, PeftCD adopts a shared-weights Siamese encoder instantiated from a VFM, into which LoRA and Adapter modules are injected as fine-tuning strategies, so that only a small number of additional parameters need to be trained for task adaptation. To better explore the potential of VFMs in change detection, we investigate two representative backbones: the Segment Anything Model v2 (SAM2), which provides strong segmentation priors, and DINOv3, a state-of-the-art self-supervised representation learner. Meanwhile, PeftCD employs a deliberately minimal and efficient decoder to highlight the representational capacity of the backbone models. Extensive experiments demonstrate that PeftCD achieves state-of-the-art performance across multiple public datasets, including SYSU-CD (IoU 73.81%), WHUCD (92.05%), MSRSCD (64.07%), MLCD (76.89%), CDD (97.01%), S2Looking (52.25%) and LEVIR-CD (85.62%), with notably precise boundary delineation and strong suppression of pseudo-changes. Overall, PeftCD achieves a favorable balance among accuracy, efficiency, and generalization, offering an efficient paradigm for adapting VFMs to practical remote sensing change detection. The code and pretrained weights are available at https://github.com/dyzy41/PeftCD.

The source code and pre-trained weights are available at https://github.com/dyzy41/PeftCD.

## Quantitative Results (Test Set Performance)

![alt text](metric.png)

## running steps

``bash install_env.sh``  
``bash run.sh``

## Citation 

 If you use this code for your research, please cite our papers.  

```
@ARTICLE{11458815,
  author={Dong, Sijun and Hu, Yuxuan and Wang, Libo and Chen, Geng and Meng, Xiaoliang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={PeftCD: Leveraging Vision Foundation Models with Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection}, 
  year={2026},
  volume={},
  number={},
  pages={1-16},
  keywords={Earth Observing System;Satellite images;Feeds;LoRa;Pixel;Communication systems;Internet;Electronic mail;Computer networks;Data communication;Remote Sensing Change Detection;Vision Foundation Models;Parameter-Efficient Fine-Tuning},
  doi={10.1109/JSTARS.2026.3679260}}
```

## Some other change detection repositories 
[ChangeCLIP](https://github.com/dyzy41/PeftCD)  
[CSDNet](https://github.com/dyzy41/CSDNet)  
[EfficientCD](https://github.com/dyzy41/mmrscd)  
[Open-CD](https://github.com/likyoo/open-cd)  
# PeftCD 使用说明

本文档补充项目的环境安装、数据准备、训练、恢复训练、单独测试以及 `main.py` 主要参数说明。代码入口为 `main.py`，示例运行脚本为 `run.sh`。

## 1. 环境安装

建议先创建独立 conda 环境：

```bash
conda create -n pl python=3.9 -y
conda activate pl
```

然后在项目根目录执行：

```bash
bash install_env.sh
```

`install_env.sh` 会安装 PyTorch 2.4.0、TorchVision 0.19.0、CUDA 11.8 对应 wheel、MMCV 2.2.0、MMEngine、Lightning 2.4.0、Albumentations、Timm、Einops、PEFT、MMSegmentation 等依赖。

如果安装后遇到 `mmsegmentation` 与 `mmcv` 的版本断言冲突，`install_env.sh` 末尾给出的处理方式是将 mmseg 的版本判断从：

```python
assert (mmcv_min_version <= mmcv_version < mmcv_max_version)
```

修改为：

```python
assert (mmcv_min_version <= mmcv_version <= mmcv_max_version)
```

## 2. 预训练权重准备

SAM2 与 DINOv3 的预训练权重通过环境变量 `PRETRAIN` 定位，需要先设置：

```bash
export PRETRAIN=/path/to/pretrain
```

推荐目录结构如下：

```text
$PRETRAIN/
  sam2/
    sam2_hiera_large.pt
  dinov3_weights/
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

DINOv3 通过 `torch.hub.load(repo_or_dir="./", model="dinov3_vitl16", source="local")` 从当前项目目录加载，因此请在项目根目录执行训练和测试命令。

## 3. 数据集准备

普通变化检测任务使用 txt 文件描述数据划分。`main.py` 默认读取：

```text
<dataset_root>/train.txt
<dataset_root>/val.txt
<dataset_root>/test.txt
```

每个 txt 文件中每一行包含三个路径，依次为第一时相图像、第二时相图像和变化标签：

```text
/path/to/A/image_001.png  /path/to/B/image_001.png  /path/to/label/image_001.png
```

标签图会以灰度图读取，并通过 `label // 255` 转成 0/1 类别，因此推荐使用 0 表示未变化、255 表示变化。

如果 `--dataset` 传入的路径不存在，代码会自动拼接环境变量 `CDPATH`：

```bash
export CDPATH=/path/to/datasets
```

例如：

```bash
python main.py --dataset SYSU-CD ...
```

会在本地没有 `SYSU-CD` 目录时，自动尝试读取：

```text
$CDPATH/SYSU-CD/train.txt
$CDPATH/SYSU-CD/val.txt
$CDPATH/SYSU-CD/test.txt
```

## 4. 训练

可以直接运行项目提供的脚本：

```bash
bash run.sh
```

`run.sh` 会先指定 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

随后依次启动四组 SYSU-CD 实验：

1. `DINO3CD + lora`
2. `DINO3CD + adapter`
3. `SAM2CD + lora`
4. `SAM2CD + adapter`

单独运行一组 DINO3CD + LoRA 实验的示例：

```bash
python main.py \
  --dataset SYSU-CD \
  --model_type cd \
  --model_arch SEED_PEFT \
  --peft_method lora \
  --model_name DINO3CD \
  --exp_name DINO3CD_DPT_SYSU_lora \
  --max_steps 15000 \
  --batch_size 16 \
  --devices 2 \
  --strategy ddp_find_unused_parameters_true \
  --accelerator gpu \
  --src_size 256 \
  --lr 0.0003 \
  --work_dirs work_dirs
```

训练文件默认保存到：

```text
<work_dirs>/<exp_name>_TrainingFiles/
```

其中包括 Lightning 日志、CSV 日志、最优 checkpoint、last checkpoint、测试预测结果和指标文件。

## 5. 恢复训练

如果训练中断，可以使用 `--resume_path` 从指定 checkpoint 继续训练：

```bash
python main.py \
  --dataset SYSU-CD \
  --model_type cd \
  --model_arch SEED_PEFT \
  --peft_method lora \
  --model_name DINO3CD \
  --exp_name DINO3CD_DPT_SYSU_lora \
  --resume_path work_dirs/DINO3CD_DPT_SYSU_lora_TrainingFiles/last.ckpt \
  --max_steps 15000 \
  --batch_size 16 \
  --devices 2 \
  --strategy ddp_find_unused_parameters_true \
  --accelerator gpu \
  --src_size 256 \
  --lr 0.0003 \
  --work_dirs work_dirs
```

## 6. 单独测试

训练模式下，`main.py` 会在 `trainer.fit(...)` 结束后自动执行：

```python
trainer.test(model, test_loader, ckpt_path="best")
```

如果只想单独测试已有 checkpoint，可以传入非 `train` 模式，并指定 `--resume_path`：

```bash
python main.py \
  --mode test_loader \
  --dataset SYSU-CD \
  --model_type cd \
  --model_arch SEED_PEFT \
  --peft_method lora \
  --model_name DINO3CD \
  --exp_name DINO3CD_DPT_SYSU_lora \
  --resume_path work_dirs/DINO3CD_DPT_SYSU_lora_TrainingFiles/best-model-xxxxxx-0.xxxx.ckpt \
  --batch_size 16 \
  --devices 1 \
  --accelerator gpu \
  --src_size 256 \
  --work_dirs work_dirs
```

注意：当前 `main.py` 的测试分支通过如下字典选择 dataloader：

```python
loader_dict = {
    "train_loader": train_loader,
    "val_loader": val_loader,
    "test_loader": test_loader,
}
```

因此单独测试时建议使用 `--mode test_loader`。如果希望命令写成 `--mode test`，需要同步修改代码中的 `loader_dict` 键名或取值逻辑。

预测图像默认保存到：

```text
<work_dirs>/<exp_name>_TrainingFiles/test_results/
```

指标文件默认保存到：

```text
<work_dirs>/<exp_name>_TrainingFiles/<exp_name>_metrics.csv
```

## 7. 主要参数说明

### 基本运行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `LEVIR-CD` | 数据集根目录或数据集名称。如果路径不存在，会拼接 `$CDPATH/<dataset>`。 |
| `--model_name` | `SAM2CD` | 具体变化检测网络，常用 `SAM2CD`、`DINO3CD`。 |
| `--model_type` | `cd` | 任务类型。`cd` 为普通变化检测；`dgcd` 为域泛化变化检测。 |
| `--model_arch` | `SEED` | Lightning 封装模型。常用 `SEED_PEFT` 启用 PEFT 训练；代码中还包含 `SEED`、`SEED_DG` 等类。 |
| `--peft_method` | `lora` | 参数高效微调方法。支持 `lora`、`adapter`，其他值会进入 IA3 配置分支。 |
| `--mode` | `train` | 运行模式。`train` 表示训练并在结束后测试；单独测试建议使用 `test_loader`。 |
| `--resume_path` | `None` | checkpoint 路径。训练时用于恢复训练；测试时用于加载指定权重。 |
| `--exp_name` | `Default` | 实验名称，同时用于日志、checkpoint 和结果目录命名。 |
| `--work_dirs` | `work_dirs` | 训练文件、日志、checkpoint 和测试结果的保存根目录。 |

### 数据加载与图像尺寸

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--batch_size` | `16` | 每张 GPU 上的 batch size。多卡训练时总 batch size 通常约为 `batch_size * GPU 数量`。 |
| `--num_workers` | `8` | DataLoader 读取进程数。Windows 或内存较小时可适当调小。 |
| `--resize_size` | `1` | 若大于 1，训练和测试都会先 resize 到该尺寸。 |
| `--src_size` | `1024` | 原图尺寸，用于判断是否需要滑窗推理。 |
| `--crop_size` | `256` | 训练随机裁剪尺寸，也是滑窗推理窗口大小。 |
| `--overlap` | `128` | 滑窗推理步长参数。代码中将其作为 stride 使用；值越小，窗口重叠越多。 |

当 `resize_size > 1` 时，代码使用 resize 流程；否则如果 `src_size > crop_size`，训练使用随机裁剪，测试使用原图并在推理时滑窗；如果 `src_size <= crop_size`，则直接使用基础变换。

### 损失与优化器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num_classes` | `2` | 输出类别数，二类变化检测一般为 2。 |
| `--pred_idx` | `0` | 多输出分支时用于选择某个输出进行推理；`SEED` 系列会对两个输出求平均。 |
| `--loss_type` | `ce` | 损失类型。`ce` 使用 `CrossEntropyLoss`，其他值会使用 `BCEWithLogitsLoss`。 |
| `--loss_weights` | `1.0 1.0` | 多输出 loss 的权重，例如 `--loss_weights 0.5 1.0`。 |
| `--lr` | `0.0003` | 初始学习率。 |
| `--min_lr` | `0.00003` | 学习率衰减下限。 |
| `--warmup` | `3000` | warmup step 数。 |
| `--optimizer` | `adamw` | `SEED_PEFT` 中支持 `adam`、`sgd`、`adamw`，其他值会使用 RMSprop。 |

### 训练进度与验证

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--max_epochs` | `120` | 最大训练 epoch 数。若 `--max_steps != -1`，代码会将 `max_epochs` 置为 `None`。 |
| `--max_steps` | `40000` | 最大训练 step 数。`run.sh` 中设置为 `15000`。若设为 `-1`，代码会根据数据量和 epoch 数估算 step。 |
| `--early_stop` | `80` | EarlyStopping 的 patience，监控指标为 `val_iou`。 |
| `--check_val_every_n_epoch` | `20` | 每多少个 epoch 验证一次；使用 `max_steps` 控制训练时会被置为 `None`。 |
| `--val_check_interval` | `1000` | 每多少个 training step 做一次验证。 |
| `--val_vis_num` | `0` | 预留的验证可视化数量参数，当前主流程中未显式使用。 |

### 日志、硬件与分布式

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--comet` / `--no-comet` | `--comet` | 是否启用 CometLogger。若不想上传日志，可加 `--no-comet`。 |
| `--save_test_results` | `test_results` | 测试预测图保存的子目录名称。 |
| `--accelerator` | `gpu` | Lightning accelerator，可按环境改为 `cpu`、`gpu` 等。 |
| `--devices` | `1` | 使用设备数。多 GPU 可设为 `2`、`4` 等，并配合 `CUDA_VISIBLE_DEVICES`。 |
| `--strategy` | `auto` | Lightning 分布式策略。多卡 PEFT 示例使用 `ddp_find_unused_parameters_true`。 |
| `--precision` | `16` | 训练精度，常用 `16` 或 `32`。 |

## 8. 常用命令模板

单卡快速调试：

```bash
export CDPATH=/path/to/datasets
export PRETRAIN=/path/to/pretrain
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --dataset SYSU-CD \
  --model_type cd \
  --model_arch SEED_PEFT \
  --model_name DINO3CD \
  --peft_method lora \
  --exp_name debug_dino_lora \
  --max_steps 100 \
  --batch_size 2 \
  --devices 1 \
  --accelerator gpu \
  --src_size 256 \
  --crop_size 256 \
  --work_dirs work_dirs \
  --no-comet
```

双卡正式训练：

```bash
export CDPATH=/path/to/datasets
export PRETRAIN=/path/to/pretrain
export CUDA_VISIBLE_DEVICES=0,1

python main.py \
  --dataset SYSU-CD \
  --model_type cd \
  --model_arch SEED_PEFT \
  --model_name SAM2CD \
  --peft_method adapter \
  --exp_name SAM2CD_DPT_SYSU_adapter \
  --max_steps 15000 \
  --batch_size 16 \
  --devices 2 \
  --strategy ddp_find_unused_parameters_true \
  --accelerator gpu \
  --src_size 256 \
  --crop_size 256 \
  --lr 0.0003 \
  --work_dirs work_dirs
```
