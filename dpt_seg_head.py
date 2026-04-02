import torch
import torch.nn as nn
import torch.nn.functional as F

class _FusionBlock(nn.Module):
    """上采样 + 残差细化 + 残差连接"""
    def __init__(self, dim):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = x + skip
        return self.act(self.res(x) + x)

class DPTSegHead(nn.Module):
    """
    DPT 风格分割解码器（同尺度多层 -> 伪多尺度金字塔 -> 级联融合 -> 预测）
    用法：
      - 3层：例如来自 ViT 的 [layer9, layer10, layer11]  -> in_ch_per_layer 长度为3
      - 4层：例如 [2,5,8,11]（推荐）                    -> in_ch_per_layer 长度为4
    """
    def __init__(self, in_ch_per_layer=[768,768,768,768], embed=256, num_classes=2, out_size=(256,256)):
        super().__init__()
        assert len(in_ch_per_layer) in (3, 4), "DPTSegHead 仅支持 3 或 4 层输入"
        self.L = len(in_ch_per_layer)
        self.out_size = out_size

        # 逐层 1x1 投影到统一 embed 维度
        self.proj = nn.ModuleList([nn.Conv2d(c, embed, 1, bias=False) for c in in_ch_per_layer])
        self.proj_bn = nn.ModuleList([nn.BatchNorm2d(embed) for _ in in_ch_per_layer])

        # 伪金字塔各层的轻量 refine
        self.refine32 = nn.Sequential(nn.Conv2d(embed, embed, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(embed), nn.GELU())
        self.refine16 = nn.Sequential(nn.Conv2d(embed, embed, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(embed), nn.GELU())
        self.refine8  = nn.Sequential(nn.Conv2d(embed, embed, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(embed), nn.GELU())
        self.refine4  = nn.Sequential(nn.Conv2d(embed, embed, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(embed), nn.GELU())

        # 用 stride=2 把最深层降到 1/32
        self.to32 = nn.Sequential(
            nn.Conv2d(embed, embed, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed), nn.GELU()
        )

        # 级联融合：32->16->8->4
        self.fuse16 = _FusionBlock(embed)
        self.fuse8  = _FusionBlock(embed)
        self.fuse4  = _FusionBlock(embed)

        # 预测头（在 1/4 尺度上预测）
        self.cls = nn.Conv2d(embed, num_classes, 1)

        # 简易 SE 门控（可选，和你原头保持“风味”一致）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed, embed//4, 1), nn.ReLU(True),
            nn.Conv2d(embed//4, embed, 1), nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, feats):
        """
        feats: List[Tensor]，每个 [B, C, H, W]，同尺度。
        请按从浅到深的顺序传入：例如 [layer9, layer10, layer11]。
        """
        assert len(feats) == self.L
        # 1) 投影到 embed
        xs = []
        for f, p, bn in zip(feats, self.proj, self.proj_bn):
            x = p(f); x = bn(x); x = F.gelu(x, approximate='tanh')
            xs.append(x)

        if self.L == 4:
            # e1(最浅 1/16)、e2(1/16)、e3(1/16)、e4(最深 1/16)
            e1, e2, e3, e4 = xs
            # 1/32
            y32 = self.refine32(self.to32(e4))
            # 1/16
            y16 = self.fuse16(y32, self.refine16(e3))
            # 1/8
            y8  = self.fuse8(y16, self.refine8(F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=False)))
            # 1/4
            y4  = self.fuse4(y8,  self.refine4(F.interpolate(e1, scale_factor=4, mode='bilinear', align_corners=False)))
        else:
            # 3层：e1(浅 1/16), e2(中 1/16), e3(深 1/16)
            e1, e2, e3 = xs
            # 1/32
            y32 = self.refine32(self.to32(e3))
            # 1/16
            y16 = self.fuse16(y32, self.refine16(e2))
            # 1/8
            y8  = self.fuse8(y16, self.refine8(F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=False)))
            # 1/4（3层时没有额外 skip，就把 y8 再上采样到 1/4 后细化）
            y4  = self.refine4(F.interpolate(y8, scale_factor=2, mode='bilinear', align_corners=False))

        # SE 门控 + 分类头
        w = self.se(y4); y4 = y4 * w
        logits_1_4 = self.cls(y4)

        if self.out_size is not None and logits_1_4.shape[-2:] != self.out_size:
            logits = F.interpolate(logits_1_4, size=self.out_size, mode='bilinear', align_corners=False)
        else:
            logits = logits_1_4
        return logits
