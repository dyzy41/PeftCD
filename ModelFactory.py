import torch
import os
import cv2
import numpy as np
from torchmetrics.classification import Accuracy, ConfusionMatrix
import lightning as L
import torch.nn as nn
from utils.metric import CM2Metric, save_metrics
import torch.optim as optim
from torch.optim.lr_scheduler import *

import torch.nn.functional as F
from SAM2CD import SAM2CD
from DINO3CD import DINO3CD


# 方法一：手动构建映射字典
_model_factory = {
    'SAM2CD': SAM2CD,
    'DINO3CD': DINO3CD

}

def build_model(name: str, *args, **kwargs):
    """
    根据 name 字符串返回对应的模型实例。
    支持传入构造函数的参数 args, kwargs。
    """
    try:
        ModelClass = _model_factory[name]
    except KeyError:
        raise ValueError(f"Unknown model name '{name}'. Available: {list(_model_factory.keys())}")
    return ModelClass(*args, **kwargs)



# 1. 定义 Lightning 模型
class BaseCD(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        if args.resize_size > 1:
            self.example_input_array = torch.randn((2, 6, args.resize_size, args.resize_size))
        else:
            self.example_input_array = torch.randn((2, 6, args.crop_size, args.crop_size))
        # define parameters
        self.save_hyperparameters(args)

        self.hyparams = args
        if self.hyparams.resize_size > 1:
            self.if_slide = False
        else:
            self.if_slide = self.hyparams.src_size > self.hyparams.crop_size
        self.save_test_results = os.path.join(self.hyparams.work_dirs, self.hyparams.exp_name+'_TrainingFiles', self.hyparams.save_test_results)

        # model training
        self.change_detection = build_model(self.hyparams.model_name)
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hyparams.num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hyparams.num_classes)
        if self.hyparams.loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.val_loss_epoch = []
        
        # prepare test output directory once
        self.test_output_dir = os.path.join(self.hyparams.work_dirs, f"{self.hyparams.exp_name}_TrainingFiles",
                                            self.hyparams.save_test_results)
        os.makedirs(self.test_output_dir, exist_ok=True)

    def forward(self, x):
        xA, xB = x[:, :3], x[:, 3:]
        out = self.change_detection(xA, xB)
        if (isinstance(out, tuple) or isinstance(out, list)) is False:
            out = [out]
        return out

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch['imgAB'], batch['lab']
        outs = self(x)
        loss = self._loss(outs, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, pathA, pathB = batch['imgAB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, val_loss_step = self._infer(x, y)
        self.val_loss_epoch.append(val_loss_step)
        self.val_confusion_matrix.update(self._logits2preds(logits), y)

    def on_validation_epoch_end(self):
        # 在所有 batch 更新完成后，compute 出整 epoch 的混淆矩阵
        cm = self.val_confusion_matrix.compute().cpu().numpy()
        metrics = CM2Metric(cm)
        val_loss_epoch = torch.mean(torch.stack(self.val_loss_epoch))
        self.log('val_loss', val_loss_epoch, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log_dict({
            'val_oa': metrics[0],
            'val_iou': metrics[4][1],
            'val_f1': metrics[5][1],
            'val_recall': metrics[6][1],
            'val_precision': metrics[7][1]
        }, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        # 重置，为下一个 epoch 准备
        self.val_confusion_matrix.reset()
        self.val_loss_epoch = []

    def test_step(self, batch: dict, batch_idx: int) -> None:
        x, y, pathA, pathB = batch['imgAB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, test_loss = self._infer(x, y)
        self.test_confusion_matrix.update(self._logits2preds(logits), y)

        pred_np = self._logits2preds(logits).cpu().numpy().astype('uint8')
        for p, mask in zip(pathA, pred_np):
            base = os.path.splitext(os.path.basename(p))[0] + '.png'
            out_path = os.path.join(self.test_output_dir, base)
            cv2.imwrite(out_path, (mask * 255).astype('uint8'))

    def on_test_epoch_end(self):
        cm = self.test_confusion_matrix.compute().cpu().numpy()
        metrics = CM2Metric(cm)
        self.log_dict({
            'test_oa': metrics[0],
            'test_iou': metrics[4][1],
            'test_f1': metrics[5][1],
            'test_recall': metrics[6][1],
            'test_precision': metrics[7][1]
        }, prog_bar=True, sync_dist=True)
        # 重置，为下一个 epoch 准备

        save_metrics(save_path=os.path.join(self.hyparams.work_dirs, self.hyparams.exp_name+'_TrainingFiles', os.path.basename(self.hyparams.exp_name)+'_metrics.csv'), metrics=metrics)

        self.test_confusion_matrix.reset()

    def _logits2preds(self, logits):
        """Convert logits to predictions."""
        if self.hyparams.loss_type == 'ce':
            preds = logits.argmax(dim=1)
        else:
            preds = torch.sigmoid(logits).round()
            preds = preds.squeeze(1)  # Remove the channel dimension for binary segmentation
        return preds

    def _loss(self, outs, y, state='train'):
        if state == 'train':
            if self.hyparams.loss_type == 'ce':
                loss = sum(w * self.criterion(o, y.long()) for w, o in zip(self.hyparams.loss_weights, outs))
            else:
                loss = sum(w * self.criterion(o, y.unsqueeze(1).float()) for w, o in zip(self.hyparams.loss_weights, outs))
        else:
            if self.hyparams.loss_type == 'ce':
                loss = self.criterion(outs, y.long())
            else:
                loss = self.criterion(outs, y.unsqueeze(1).float())
        return loss

    def _infer(self, x, y):
        """Run either sliding-window or single-pass inference."""

        if self.if_slide:
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss =  self._loss(outs, y)
            logits = outs[self.hyparams.pred_idx]
            return logits, val_loss

    def _slide_inference(self, inputs, labels):
        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                outs = self(crop_img)
                crop_seg_logit = outs[self.hyparams.pred_idx]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        val_loss = self._loss(seg_logits, labels, state='val')

        return seg_logits, val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hyparams.lr, weight_decay=1e-4)
        
        def lr_lambda(step: int) -> float:
            warmup = self.hyparams.warmup
            power = 3.0
            if step < warmup:
                raw = float(step) / float(max(1, warmup))
            else:
                progress = float(step - warmup) / float(max(1, self.hyparams.max_steps - warmup))
                raw = max(0.0, (1.0 - progress) ** power)
            min_factor = self.hyparams.min_lr / self.hyparams.lr
            return max(raw, min_factor)
        
        scheduler = LambdaLR(optimizer, lr_lambda) 

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                # "monitor": "val_iou",
                "strict": False,
                "frequency": 1,
                "name": None
            },
        }


class SEED(BaseCD):
    """
    SEED Lightning 模型：
    - 继承自 BaseCD，复用数据加载、训练/验证/测试流程
    - 只需要在 __init__ 中替换 change_detection 网络即可
    """
    def __init__(self, args):
        # 调用父类，完成超参保存、confusion matrix、criterion 等初始化
        super().__init__(args)

    def _loss(self, outs, y, state='train'):
        """Compute the loss for the current batch."""
        if state == 'train':
            loss = sum(w * self.criterion(o, y.long())
                        for w, o in zip(self.hyparams.loss_weights, outs))
            loss = loss/2.0
        else:
            loss = self.criterion(outs, y.long())
        return loss

    def _infer(self, x, y):
        """Run either sliding-window or single-pass inference."""

        if self.if_slide and self.hyparams.model_type != 'dgcd':
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss =  self._loss(outs, y)
            logits = (outs[0]+outs[1])/2.0
            return logits, val_loss

    def _slide_inference(self, inputs, labels):

        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                outs = self(crop_img)
                crop_seg_logit = (outs[0]+outs[1])/2.0

                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        val_loss = self._loss(seg_logits, labels, state='val')

        return seg_logits, val_loss


class SEED_PEFT(SEED):
    """
    SEED Lightning 模型：
    - 继承自 BaseCD，复用数据加载、训练/验证/测试流程
    - 只需要在 __init__ 中替换 change_detection 网络即可
    """
    def __init__(self, args):
        # 调用父类，完成超参保存、confusion matrix、criterion 等初始化
        super().__init__(args)
        
        self.change_detection = build_model(self.hyparams.model_name, peft_method=self.hyparams.peft_method)

    def configure_optimizers(self):
        # ==================== 主要修改在这里 ====================
        # 筛选出所有 requires_grad = True 的参数
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        # 优化器只接收可训练的参数
        if self.hyparams.optimizer == 'adam':
            optimizer = optim.Adam(trainable_params, lr=self.hyparams.lr, weight_decay=1e-4)
        elif self.hyparams.optimizer == 'sgd':
            optimizer = optim.SGD(trainable_params, lr=self.hyparams.lr, momentum=0.9, nesterov=True)
        elif self.hyparams.optimizer == 'adamw':
            optimizer = optim.AdamW(trainable_params, lr=self.hyparams.lr, weight_decay=1e-4)
        else:
            optimizer = optim.RMSprop(trainable_params, lr=self.hyparams.lr, alpha=0.99)
        # ========================================================

        # 学习率调度器的逻辑完全保持不变
        def lr_lambda(step: int) -> float:
            warmup = self.hyparams.warmup
            power = 3.0
            if step < warmup:
                raw = float(step) / float(max(1, warmup))
            else:
                progress = float(step - warmup) / float(max(1, self.hyparams.max_steps - warmup))
                raw = max(0.0, (1.0 - progress) ** power)
            min_factor = self.hyparams.min_lr / self.hyparams.lr
            return max(raw, min_factor)
        
        scheduler = LambdaLR(optimizer, lr_lambda) 

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
                "frequency": 1,
                "name": None
            },
        }

    # def configure_optimizers(self):
    #     # ==================== 主要修改在这里 ====================
    #     # 1. 将可训练参数分组
    #     lora_params = []
    #     other_trainable_params = []
        
    #     # 我们通过参数名来区分它们
    #     # 假设LoRA参数都在encoder中，其他可训练参数（FPN, decoder等）不在
    #     for name, param in self.named_parameters():
    #         if not param.requires_grad:
    #             continue
            
    #         if 'encoder.' in name: # 假设LoRA参数都在encoder模块里
    #             lora_params.append(param)
    #         else:
    #             other_trainable_params.append(param)

    #     # 2. 为不同组设置不同的学习率
    #     # 主学习率用于LoRA参数
    #     main_lr = self.hyparams.lr
    #     # 其他参数使用一个较小的学习率，例如主学习率的1/10
    #     secondary_lr = self.hyparams.lr / 10.0
        
    #     print(f"LoRA参数数量: {len(lora_params)}")
    #     print(f"其他可训练参数数量: {len(other_trainable_params)}")
        
    #     param_groups = [
    #         {'params': lora_params, 'lr': main_lr},
    #         {'params': other_trainable_params, 'lr': secondary_lr}
    #     ]

    #     # 优化器接收参数组
    #     optimizer = optim.AdamW(param_groups, lr=self.hyparams.lr, weight_decay=1e-4)
    #     # ========================================================
        
    #     # 学习率调度器的逻辑完全保持不变
    #     # LambdaLR会自动按比例缩放每个参数组的学习率
    #     def lr_lambda(step: int) -> float:
    #         warmup = self.hyparams.warmup
    #         power = 3.0
    #         if step < warmup:
    #             raw = float(step) / float(max(1, warmup))
    #         else:
    #             progress = float(step - warmup) / float(max(1, self.hyparams.max_steps - warmup))
    #             raw = max(0.0, (1.0 - progress) ** power)
    #         min_factor = self.hyparams.min_lr / self.hyparams.lr
    #         return max(raw, min_factor)
        
    #     scheduler = LambdaLR(optimizer, lr_lambda) 

    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #             "strict": False,
    #             "frequency": 1,
    #             "name": None
    #         },
    #     }



class SEED_DG(BaseCD):
    """
    SEED Lightning 模型：
    - 继承自 BaseCD，复用数据加载、训练/验证/测试流程
    - 只需要在 __init__ 中替换 change_detection 网络即可
    """
    def __init__(self, args):
        # 调用父类，完成超参保存、confusion matrix、criterion 等初始化
        super().__init__(args)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch['imgAB'], batch['lab']
        outs = self(x)
        outs2 = self(x)
        loss = self._loss(outs, y)
        loss2 = self._loss(outs2, y)
        pixel_loss = (loss + loss2) / 2.0

        consistency_loss = (kl_loss(outs[0], outs2[0]) + kl_loss(outs[1], outs2[1])) / 2.0

        consistency_weight = 0.05
        rampup_duration = 1000

        progress = self.global_step / rampup_duration
        current_weight = consistency_weight * min(1.0, progress)

        # 4. 计算总损失
        total_loss = pixel_loss + current_weight * consistency_loss

        # 记录训练损失
        self.log('train_pixel_loss', pixel_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_consistency_loss', consistency_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def _loss(self, outs, y, state='train'):
        """Compute the loss for the current batch."""
        if state == 'train':
            loss = sum(w * self.criterion(o, y.long())
                        for w, o in zip(self.hyparams.loss_weights, outs))
        else:
            loss = self.criterion(outs, y.long())
        return loss/2.0

    def _infer(self, x, y):
        """Run either sliding-window or single-pass inference."""

        if self.if_slide and self.hyparams.model_type != 'dgcd':
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss =  self._loss(outs, y)
            logits = (outs[0]+outs[1])/2.0
            return logits, val_loss

    def _slide_inference(self, inputs, labels):

        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                outs = self(crop_img)
                crop_seg_logit = (outs[0]+outs[1])/2.0

                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        val_loss = self._loss(seg_logits, labels, state='val')

        return seg_logits, val_loss
    

class SEED_DIY(SEED):
    """
    SEED Lightning 模型：
    - 继承自 BaseCD，复用数据加载、训练/验证/测试流程
    - 只需要在 __init__ 中替换 change_detection 网络即可
    """
    def __init__(self, args):
        # 调用父类，完成超参保存、confusion matrix、criterion 等初始化
        super().__init__(args)

    def _loss(self, outs, y, state='train'):
        """Compute the loss for the current batch."""
        if state == 'train':
            loss = sum(w * self.criterion(o, y.long())
                        for w, o in zip(self.hyparams.loss_weights, [outs[0], outs[1]]))
            loss = loss/2.0
            if len(outs) > 2:
                xA_list, xB_list = outs[2], outs[3]
                for xA, xB in zip(xA_list, xB_list):
                    loss += F.mse_loss(xA, xB)*0.1
        else:
            loss = self.criterion(outs, y.long())
        return loss


class SEED_Test(SEED):
    """
    SEED Lightning 模型：
    - 继承自 BaseCD，复用数据加载、训练/验证/测试流程
    - 只需要在 __init__ 中替换 change_detection 网络即可
    """
    def __init__(self, args):
        # 调用父类，完成超参保存、confusion matrix、criterion 等初始化
        super().__init__(args)

    def _loss(self, outs, y, state='train'):
        """Compute the loss for the current batch."""
        if state == 'train':
            outs_logits, coral_features = outs[0], outs[1]
            loss = 0.00
            cd_loss = sum(w * self.criterion(o, y.long())
                        for w, o in zip(self.hyparams.loss_weights, outs_logits))
            loss += cd_loss/2.0
            loss += coral_loss(coral_features[0], coral_features[1]) * 0.1
        else:
            loss = self.criterion(outs, y.long())
        return loss
