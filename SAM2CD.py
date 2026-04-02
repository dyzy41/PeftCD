import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import os
from utils.exchange import FeatureExchanger, ExchangeType
from mmseg.registry import MODELS
from utils.decode_block import *
import math

from peft import get_peft_model, LoraConfig
from peft import IA3Model, IA3Config


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class SAM2CD(nn.Module):
    def __init__(self, checkpoint_path=None, peft_method='lora') -> None:
        super(SAM2CD, self).__init__()    
        # model_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_configs/sam2_hiera_l.yaml")
        model_cfg = 'sam2_hiera_l'
        checkpoint_path = os.path.join(os.getenv('PRETRAIN'), f'sam2/sam2_hiera_large.pt')
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        self.get_encoder_peft(peft_method)

        # print("Verifying which parameters are trainable:")
        # for name, param in self.encoder.named_parameters(): # 遍历整个 SAM2CD_LoRA 模型
        #     if param.requires_grad:
        #         print(f"Trainable parameter: {name} | size: {param.size()}")

        FPN_DICT = {'type': 'FPN', 'in_channels': [144, 288, 576, 1152], 'out_channels': 256, 'num_outs': 4}
        self.fpn = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels'])
        )

        self.ex_func = FeatureExchanger(training=self.training)

        self.decode_conv = nn.Sequential(
            nn.Conv2d(FPN_DICT['out_channels'], FPN_DICT['out_channels'], kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(FPN_DICT['out_channels'], 2, kernel_size=1)

    def get_encoder_peft(self, peft_method):
        if peft_method=='adapter':
            for param in self.encoder.parameters():
                param.requires_grad = False
            blocks = []
            for block in self.encoder.blocks:
                blocks.append(
                    Adapter(block)
                )
            self.encoder.blocks = nn.Sequential(
                *blocks
            )
        elif peft_method=='lora':

            lora_config = LoraConfig(
                r=8,  # 低秩矩阵的秩
                lora_alpha=32,  # Alpha值，控制适应度
                target_modules=["qkv"],  # 目标模块，可以是qkv等
                lora_dropout=0.1,  # LoRA的dropout比例
            )

            # 在encoder中应用LoRA
            self.encoder = get_peft_model(self.encoder, lora_config)
            # self.encoder.print_trainable_parameters()  # 打印可训练参数
        else:
            # 配置IA3
            IA3_config = IA3Config(
                target_modules=["qkv"],
                feedforward_modules=[],
            )

            # 在encoder中应用LoRA
            self.encoder = get_peft_model(self.encoder, IA3_config)
            # self.encoder.print_trainable_parameters()  # 打印可训练参数

    def decode_stage(self, feature_list):
        x1, x2, x3, x4 = feature_list
        x4 = self.decode_layersA[4](x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = x3 + x4
        x3 = self.decode_layersA[3](x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = x2 + x3
        x2 = self.decode_layersA[2](x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = x1 + x2
        x1 = self.decode_layersA[1](x1)
        return x1

    def decode_head(self, x, out_size=None):
        x = self.decode_conv(x)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        x = self.conv_seg(x)
        return x

    def forward(self, xA, xB, labels=None):
        out_size = xA.shape[2:]
        xA_list = self.encoder(xA)
        xB_list = self.encoder(xB)

        xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=ExchangeType.LAYER, thresh=0.5)

        xA_list = self.fpn(xA_list)
        xB_list = self.fpn(xB_list)

        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)

        outA = F.interpolate(outA, scale_factor=2, mode='bilinear', align_corners=False)
        outB = F.interpolate(outB, scale_factor=2, mode='bilinear', align_corners=False)

        outA = self.decode_head(outA, out_size)
        outB = self.decode_head(outB, out_size)
        change_maps = [outA, outB]

        return change_maps



if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2CD().cuda()
        x = torch.randn(2, 3, 256, 256).cuda()
        out, out1, out2 = model(x, x)
        print(out.shape, out1.shape, out2.shape)