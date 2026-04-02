import torch
import torch.nn as nn
import os
from utils.exchange import FeatureExchanger, ExchangeType
from utils.decode_block import *

from peft import get_peft_model, LoraConfig
from peft import IA3Model, IA3Config
from VFMDecoder import SSBiFPNDecoder


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

    def forward(self, x, rope_sincos=None):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped, rope_sincos)
        return net
    

class DINO3CD(nn.Module):
    def __init__(self, checkpoint_path=None, peft_method='adapter') -> None:
        super(DINO3CD, self).__init__()    
        dinov3_weight_path=os.path.join(os.getenv('PRETRAIN'), 'dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        dinov3_local_path="./"
        self.img_size = 256
        self.peft_method = peft_method
        dino = torch.hub.load(
            repo_or_dir=dinov3_local_path,
            model="dinov3_vitl16",
            source="local",
            pretrained=False,
            trust_repo=True
        )
        if dinov3_weight_path:
            checkpoint = torch.load(dinov3_weight_path, map_location='cpu')
            dino.load_state_dict(checkpoint, strict=True)
            print("✓ Local weights successfully loaded")

        self.encoder = dino
        self.get_encoder_peft(peft_method)
        
        self.ex_func = FeatureExchanger(training=self.training)
        self.decoder = SSBiFPNDecoder(in_chs=[1024, 1024, 1024, 1024], out_ch=2)
        self.inter_layers = 4

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


    def channel_exchange(self, x1: torch.Tensor, x2: torch.Tensor, p: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        _, C, _, _ = x1.shape
        mask = (torch.arange(C, device=x1.device) % p == 0).view(1, C, 1, 1)
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    def forward(self, xA, xB, labels=None):
        LAYERS = [5,11,17,23] # 0-based
        outsA = self.encoder.get_intermediate_layers(xA, n=LAYERS, reshape=True, norm=True)
        outsB = self.encoder.get_intermediate_layers(xB, n=LAYERS, reshape=True, norm=True)
        
        # outsA = self.encoder.get_intermediate_layers(
        #     xA, n=self.inter_layers, reshape=True
        # )
        # outsB = self.encoder.get_intermediate_layers(
        #     xB, n=self.inter_layers, reshape=True
        # )

        outsA = list(outsA); outsB = list(outsB)

        outsA, outsB = self.ex_func.exchange(outsA, outsB, mode=ExchangeType.LAYER, thresh=0.5)
        outA = self.decoder(outsA)
        outB = self.decoder(outsB)

        return [outA, outB]



if __name__ == "__main__":
    with torch.no_grad():
        model = DINO3CD().cuda()
        x = torch.randn(2, 3, 256, 256).cuda()
        out1, out2 = model(x, x)
        print(out1.shape, out2.shape)