from torch.utils.data import DataLoader
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning import Trainer
import os
from utils.dataset import CDTXTDataset, DGCDTXTDataset  # 假设数据集定义在dataset.py文件中
from lightning.pytorch.loggers import CSVLogger
import argparse
import ModelFactory
from lightning.pytorch.callbacks import ModelSummary
from utils.transforms import get_best_model_checkpoint, define_transforms


def check_trainable_parameters(model):
    print("=" * 80)
    print(f"{'Parameter Name':40s} {'Shape':30s} {'Trainable'}")
    print("=" * 80)
    for name, param in model.named_parameters():
        print(f"{name:40s} {str(list(param.shape)):30s} {param.requires_grad}")
    print("=" * 80)

    # 统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters: {frozen_params}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model for change detection')

    # ===== 1. 基本与运行模式 =====
    parser.add_argument('--dataset',        type=str, default='LEVIR-CD', help='Path to dataset root')
    parser.add_argument('--model_name',     type=str, default='SAM2CD', help='Model name')
    parser.add_argument('--model_type',     type=str, default='cd', help='Model type (change detection/domain generalization)')
    parser.add_argument('--model_arch',     type=str, default='SEED', help='Model Architecture')
    parser.add_argument('--peft_method',    type=str, default='lora', help='Model Architecture with PEFT')

    parser.add_argument('--mode',           type=str, default='train', help='Mode of the program (train/test)')
    parser.add_argument('--resume_path',    type=str, default=None,  help='Path to resume from checkpoint')
    parser.add_argument('--exp_name',       type=str, default='Default', help='Experiment name')
    parser.add_argument('--work_dirs',       type=str, default='work_dirs', help='Working directory for saving results')

    # ===== 2. 数据加载与预处理 =====
    parser.add_argument('--batch_size',     type=int, default=16,   help='Batch size for training')
    parser.add_argument('--num_workers',    type=int, default=8,    help='Number of workers for data loading')
    parser.add_argument('--resize_size',    type=int, default=1,  help='Resize size for input images')
    parser.add_argument('--src_size',       type=int, default=1024,  help='Source size for input images')
    parser.add_argument('--crop_size',      type=int, default=256,  help='Crop size for input images')
    parser.add_argument('--overlap',        type=int, default=128,  help='Overlap size for sliding window')

    # ===== 3. 模型输出与类别 =====
    parser.add_argument('--pred_idx',       type=int, default=0,    help='GPU ID to use / index of output branch')
    parser.add_argument('--num_classes',    type=int, default=2,    help='Number of classes for model output')

    # ===== 4. 损失与优化超参 =====
    parser.add_argument('--loss_type',      type=str,   default='ce',    help='Loss type (bce/ce/focal)')
    parser.add_argument('--loss_weights',   type=float, nargs='+', default=[1.0, 1.0], help='各个 loss 的权重，例: --loss_weights 0.5 1.0 2.0')
    parser.add_argument('--lr',             type=float, default=0.0003,  help='Learning rate')
    parser.add_argument('--min_lr',         type=float, default=0.00003,  help='Minimum learning rate')
    parser.add_argument('--warmup',         type=int,   default=3000,  help='Number of steps for warmup')
    parser.add_argument('--optimizer',      type=str,   default='adamw',  help='choose optimizer')


    # ===== 5. 训练进度控制 =====
    parser.add_argument('--max_epochs',             type=int, default=120,    help='Number of epochs to train (-1 for unlimited)')
    parser.add_argument('--max_steps',              type=int, default=40000, help='Number of steps to train')
    parser.add_argument('--early_stop',             type=int, default=80,    help='Patience for early stopping')

    # ===== 6. 验证与可视化 =====
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20,  help='Check validation every n epochs')
    parser.add_argument('--val_check_interval',      type=int, default=1000,   help='Check validation every n iterations')
    parser.add_argument('--val_vis_num',             type=int, default=0,     help='Number of validation images to visualize')

    # ===== 7. 日志与结果保存 =====
    parser.add_argument('--comet',            action=argparse.BooleanOptionalAction, default=True,  help='Use Comet logger')
    parser.add_argument('--save_test_results', type=str, default='test_results', help='Path to save test results')

    # ===== 8. 分布式与硬件配置 =====
    parser.add_argument('--accelerator',  type=str, default='gpu',   help='Accelerator for training')
    parser.add_argument('--devices',      type=str, default=1,     help='Number of devices for training')
    parser.add_argument('--strategy',     type=str, default='auto',  help='Strategy for distributed training')
    parser.add_argument('--precision',    type=int, default=16,      help='Precision for training (16 or 32)')

    args = parser.parse_args()

    if os.path.exists(args.dataset) is False:
        args.dataset = os.path.join(os.environ.get("CDPATH"), args.dataset)
    
    if args.max_steps != -1:
        args.max_epochs = None
        args.check_val_every_n_epoch = None

    for name, value in vars(args).items():
        print(f"{name}: {value}")

    train_transform_dict, test_transform_dict = define_transforms(crop_size=args.crop_size, resize_size=args.resize_size)

    if args.resize_size > 1:
        train_transform = train_transform_dict['resize']
        test_transform = test_transform_dict['resize']
    else:
        if args.src_size > args.crop_size:
            train_transform = train_transform_dict['crop']
            test_transform = test_transform_dict['base']
        else:
            train_transform = train_transform_dict['base']
            test_transform = test_transform_dict['base']

    print('Train Data Augmentation Information:')
    for idx, aug in enumerate(train_transform.transforms):
        print(f'  [{idx}] {aug}')

    print('Test Data Augmentation Information:')
    for idx, aug in enumerate(test_transform.transforms):
        print(f'  [{idx}] {aug}')
    
    if args.model_type == 'cd':
        train_dataset = CDTXTDataset(os.path.join(args.dataset, 'train.txt'), transform=train_transform)
        val_dataset = CDTXTDataset(os.path.join(args.dataset, 'val.txt'), transform=test_transform)
        test_dataset = CDTXTDataset(os.path.join(args.dataset, 'test.txt'), transform=test_transform)
    elif args.model_type == 'dgcd':
        train_dataset = DGCDTXTDataset(train_domain=['WaterCDPNG', 'PX-CLCD', 'WHUCD', 'LEVIR-CD'], split='train', transform=train_transform)
        val_dataset = DGCDTXTDataset(train_domain=['WaterCDPNG', 'PX-CLCD', 'WHUCD', 'LEVIR-CD'], split='val', transform=test_transform)
        test_dataset = DGCDTXTDataset(train_domain=['WaterCDPNG', 'PX-CLCD', 'WHUCD', 'LEVIR-CD'], split='test', transform=test_transform)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # iters*num_gpus*batch_size=epochs*num_images
    num_gpus = torch.cuda.device_count()
    if args.max_steps == -1:
        args.max_steps = (len(train_dataset) * args.max_epochs *1.00 / args.batch_size) / num_gpus + 100

    checkpoint_callback = ModelCheckpoint(
        monitor='val_iou',          # 监控的指标
        mode='max',                 # 取最大值时的权重
        save_top_k=3,               # 最多保留 5 个最优模型
        dirpath=os.path.join(args.work_dirs, args.exp_name+'_TrainingFiles'),     # 保存目录
        filename='best-model-{step:06d}-{val_iou:.4f}',  # 文件名格式
        save_last=True,          # 保存最后一个模型
        verbose=True,               # 输出日志
    )
    early_stop_callback = EarlyStopping(
                        monitor='val_iou',
                        patience=args.early_stop, 
                        verbose=True, 
                        mode='max'
                    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    csv_logger = CSVLogger(os.path.join(args.work_dirs, args.exp_name+'_TrainingFiles'), name=args.exp_name)

    if args.comet:
        from lightning.pytorch.loggers import CometLogger
        # Comet logger
        comet_logger = CometLogger(
            api_key="8nMLAmoZ0ZHRX6M19vtWa0J0T",
            workspace="dyzy41",
            project_name="change-detection",
            experiment_name=args.exp_name
        )
        running_logger = [comet_logger, csv_logger]
    else:
        running_logger = [csv_logger]

    if args.mode == 'train':
        trainer = Trainer(
            strategy=args.strategy,
            devices=args.devices,
            accelerator=args.accelerator,
            precision=args.precision,
            max_epochs=args.max_epochs,
            max_steps=args.max_steps,
            logger=running_logger,
            log_every_n_steps=2,
            gradient_clip_val=1.0,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            val_check_interval = args.val_check_interval,
            callbacks=[early_stop_callback, checkpoint_callback, RichProgressBar(leave=True), lr_monitor, ModelSummary(max_depth=3)],
        )

        model = getattr(ModelFactory, args.model_arch)(args)

        check_trainable_parameters(model)

        if args.resume_path is not None:
            print(f"Resume Training With Model From {args.resume_path}")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_path)
        else:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        trainer.test(model, test_loader, ckpt_path='best')
    else:
        if args.resume_path is not None:
            best_checkpoint_path = args.resume_path
        else:
            best_checkpoint_path = get_best_model_checkpoint(os.path.join(args.work_dirs, args.exp_name+'_TrainingFiles'))

        model = getattr(ModelFactory, args.model_arch).load_from_checkpoint(best_checkpoint_path, args=args)

        trainer = Trainer(
            strategy='auto',
            devices=1,
            num_nodes=1,
            accelerator=args.accelerator,
            precision=args.precision,
            callbacks=[RichProgressBar()]
        )
        loader_dict = {
            "train_loader": train_loader,
            "val_loader":   val_loader,
            "test_loader":  test_loader,
        }
        trainer.test(
            model=model,
            dataloaders=loader_dict[args.mode],
        )



