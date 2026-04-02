import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1BNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

class DWConvBNAct(nn.Sequential):
    def __init__(self, ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(ch, ch, k, s, p, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU()
        )

class ASPP(nn.Module):
    def __init__(self, ch, rates=(1, 2, 4, 8), out_ch=None):
        super().__init__()
        out_ch = out_ch or ch
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=r, dilation=r, groups=ch, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
            ) for r in rates
        ])
        self.proj = Conv1x1BNAct(ch * len(rates), out_ch)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.proj(x)

class DepthAttnFuse(nn.Module):
    """对同分辨率的多层特征做【空间可变】深度注意力融合"""
    def __init__(self, in_chs, mid_ch=256):
        super().__init__()
        self.proj = nn.ModuleList([Conv1x1BNAct(c, mid_ch) for c in in_chs])
        self.score = nn.ModuleList([nn.Conv2d(mid_ch, 1, 1) for _ in in_chs])
        self.post = DWConvBNAct(mid_ch)

    def forward(self, feats):
        # feats: list of [F1, F2, F3, F4], each (B,Ci,H,W), 同分辨率(1/16)
        xs = [p(f) for p, f in zip(self.proj, feats)]      # -> (B,mid,H,W)
        S = torch.stack([torch.sigmoid(h(x)) for h, x in zip(self.score, xs)], dim=1)  # (B,4,1,H,W)
        A = torch.softmax(S, dim=1)                        # 归一化到层维度
        X = torch.stack(xs, dim=1)                         # (B,4,mid,H,W)
        fused = (A * X).sum(dim=1)                         # (B,mid,H,W)
        return self.post(fused)                            # 平滑一下

class SSBiFPNDecoder(nn.Module):
    def __init__(self, in_chs=(384, 384, 768, 1024), mid_ch=256, out_ch=1):
        super().__init__()
        self.fuse = DepthAttnFuse(in_chs, mid_ch)          # 同尺度多层融合
        self.context = ASPP(mid_ch, rates=(1,2,4,8), out_ch=mid_ch)  # 感受野金字塔

        # 细节回注（1/4 分支，可换成 |A-B|、梯度等作为输入）
        self.detail_in = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),   # 1/2
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),   # 1/4
        )
        self.detail_gate = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 1), nn.Sigmoid())

        # 上采样头：1/16 -> 1/8 -> 1/4 -> 1/1
        up = (lambda: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.up1 = up(); self.up2 = up(); self.up3 = up()
        self.refine1 = DWConvBNAct(mid_ch)
        self.refine2 = DWConvBNAct(mid_ch)
        self.refine3 = DWConvBNAct(mid_ch)

        self.head = nn.Sequential(
            DWConvBNAct(mid_ch),
            nn.Conv2d(mid_ch, out_ch, 1)
        )

    def forward(self, feats, img=None):
        """
        feats: [F1,F2,F3,F4]  # 全是 1/16 的特征图
        img: 原图 (B,3,H,W) （可选；变化检测可传 concat(A,B,|A-B|) 的卷积结果）
        """
        x = self.fuse(feats)           # (B,mid,H/16,W/16)
        x = self.context(x)            # 提升全局/多尺度上下文

        x = self.up1(x); x = self.refine1(x)   # -> 1/8
        x = self.up2(x); x = self.refine2(x)   # -> 1/4

        if img is not None:
            d = self.detail_in(img)                         # 1/4
            g = self.detail_gate(F.adaptive_avg_pool2d(x, 1))
            x = x + g * d                                   # 门控细节回注

        x = self.refine3(x)           # 1/4
        x = self.up3(x)               # 1/2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 1/1
        return self.head(x)
