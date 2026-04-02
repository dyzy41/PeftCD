# feature_exchange.py

import random
import torch
from enum import Enum
from typing import List, Tuple, Optional

class ExchangeType(str, Enum):
    LAYER         = 'le'
    RAND_LAYER    = 'rle'
    CHANNEL       = 'ce'
    RAND_CHANNEL  = 'rce'
    SPATIAL       = 'se'
    RAND_SPATIAL  = 'rse'

class FeatureExchanger:
    """
    支持多种特征交换操作：
      - LAYER/RAND_LAYER: 对整个特征列表进行层级交换
      - CHANNEL/RAND_CHANNEL: 对列表中指定张量的通道做交换
      - SPATIAL/RAND_SPATIAL: 对列表中指定张量的空间维度做交换
    训练时随机交换，推理时固定交换。

    参数说明：
      thresh: 随机层交换的概率阈值
      p: 通道/空间交换的间隔或概率参数
      ratio: 随机通道交换的比例
      layers: 可选层索引列表，指定对哪些层(feature)执行 channel 或 spatial 交换，
              为 None 时默认对所有层进行操作。
    """

    def __init__(self, training: bool = True):
        self.training = training

    @staticmethod
    def layer_exchange(x: List[torch.Tensor], y: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        for i in range(0, len(x), 2):
            x[i], y[i] = y[i], x[i]
        return x, y

    @staticmethod
    def random_layer_exchange(x: List[torch.Tensor], y: List[torch.Tensor], thresh: float = 0.5) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        for i in range(len(x)):
            if torch.rand(1).item() < thresh:
                x[i], y[i] = y[i], x[i]
        return x, y

    @staticmethod
    def channel_exchange(x1: torch.Tensor, x2: torch.Tensor, p: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        _, C, _, _ = x1.shape
        mask = (torch.arange(C, device=x1.device) % p == 0).view(1, C, 1, 1)
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    @staticmethod
    def random_channel_exchange(x1: torch.Tensor, x2: torch.Tensor, thresh: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对每一个通道，以 thresh 的概率随机决定是否进行交换。
        """
        # 1. 获取通道数 C
        _, C, _, _ = x1.shape

        # 2. 创建随机掩码 (核心修改)
        #    生成 C 个 [0, 1) 之间的随机数，
        #    直接与 thresh 比较，得到一个布尔类型的掩码。
        #    掩码中每个元素为 True 的概率即为 thresh。
        mask = torch.rand(C, device=x1.device) < thresh
        
        # 3. 调整视图以进行广播 (shape: [1, C, 1, 1])
        mask = mask.view(1, C, 1, 1)

        # 4. 使用 torch.where 执行交换，代码更清晰
        #    当 mask 中元素为 True 时，out1 从 x2 取值，否则从 x1 取值。
        out1 = torch.where(mask, x2, x1)
        #    当 mask 中元素为 True 时，out2 从 x1 取值，否则从 x2 取值。
        out2 = torch.where(mask, x1, x2)
        
        return out1, out2

    @staticmethod
    def spatial_exchange(x1: torch.Tensor, x2: torch.Tensor, p: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        以固定的间隔 p，随机地在行或列上交换特征。
        """
        _, _, H, W = x1.shape
        
        # 随机选择是按行交换还是按列交换
        if random.random() < 0.5:
            # --- 按行交换 (Row Exchange) ---
            # 1. 沿着高度 H 创建掩码
            mask = (torch.arange(H, device=x1.device) % p == 0)
            # 2. 调整视图，使其能够广播到每一列 (shape: [1, 1, H, 1])
            mask = mask.view(1, 1, H, 1)
        else:
            # --- 按列交换 (Column Exchange) ---
            # 1. 沿着宽度 W 创建掩码 (原始逻辑)
            mask = (torch.arange(W, device=x1.device) % p == 0)
            # 2. 调整视图，使其能够广播到每一行 (shape: [1, 1, 1, W])
            mask = mask.view(1, 1, 1, W)
            
        # torch.where 会根据掩码的形状自动进行广播，无需修改
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    @staticmethod
    def random_spatial_exchange(x1: torch.Tensor, x2: torch.Tensor, thresh: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        以指定的概率 thresh，随机地在行或列上交换特征。
        """
        _, _, H, W = x1.shape
        
        # 随机选择是按行交换还是按列交换
        if random.random() < 0.5:
            # --- 按行随机交换 (Random Row Exchange) ---
            # 1. 沿着高度 H 创建随机掩码
            mask = (torch.rand(H, device=x1.device) < thresh)
            # 2. 调整视图以进行行广播
            mask = mask.view(1, 1, H, 1)
        else:
            # --- 按列随机交换 (Random Column Exchange) ---
            # 1. 沿着宽度 W 创建随机掩码 (原始逻辑)
            mask = (torch.rand(W, device=x1.device) < thresh)
            # 2. 调整视图以进行列广播
            mask = mask.view(1, 1, 1, W)
        
        # 应用交换
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    def exchange(
        self,
        featA: List[torch.Tensor],
        featB: List[torch.Tensor],
        mode: ExchangeType = ExchangeType.LAYER,
        thresh: float = 0.5,
        p: int = 2,
        layers: Optional[List[int]] = [2, 3]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        执行指定模式的交换操作，支持层级、通道和空间三种交换。

        参数：
          featA, featB: 待交换的特征列表
          mode: 交换模式
          thresh: 随机层交换阈值
          p: 通道/空间交换参数 (步长或概率)
          layers: 层索引列表，None 则对所有层操作
        """
        if mode == ExchangeType.LAYER:
            return self.layer_exchange(featA, featB)

        if mode == ExchangeType.RAND_LAYER:
            return self.random_layer_exchange(featA, featB, thresh) if self.training else self.layer_exchange(featA, featB)

        # 确定应用 channel/spatial 的层索引
        layer_ids = layers if layers is not None else list(range(len(featA)))

        if mode == ExchangeType.CHANNEL or mode == ExchangeType.RAND_CHANNEL:
            for i in layer_ids:
                if mode == ExchangeType.RAND_CHANNEL and self.training:
                    featA[i], featB[i] = self.random_channel_exchange(featA[i], featB[i], thresh)
                else:
                    featA[i], featB[i] = self.channel_exchange(featA[i], featB[i], p)
            return featA, featB

        if mode == ExchangeType.SPATIAL or mode == ExchangeType.RAND_SPATIAL:
            for i in layer_ids:
                if mode == ExchangeType.RAND_SPATIAL and self.training:
                    featA[i], featB[i] = self.random_spatial_exchange(featA[i], featB[i], thresh)
                else:
                    featA[i], featB[i] = self.spatial_exchange(featA[i], featB[i], p)
            return featA, featB

        # 默认层级交换
        return self.layer_exchange(featA, featB)


if __name__ == '__main__':
    # 简单测试
    A = [torch.randn(1,16,32,32) for _ in range(5)]
    B = [torch.randn_like(t) for t in A]
    exch = FeatureExchanger(training=True)
    # 仅对第2和第3层执行通道交换示例
    A2, B2 = exch.exchange(A, B, mode=ExchangeType.CHANNEL, p=2, layers=[2, 3])
    print([t.shape for t in A2], [t.shape for t in B2])
