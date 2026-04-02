import torch
import torch.nn.functional as F

def coral_loss(Fs: torch.Tensor, Ft: torch.Tensor) -> torch.Tensor:
    """
    Deep CORAL 损失
    Args:
        Fs: 源域特征，shape = (N, C, H, W)
        Ft: 目标域特征，shape = (N, C, H, W)
    Returns:
        scalar CORAL loss
    """
    N, C, H, W = Fs.size()
    # 1) 将特征图变形为 (C, N*H*W)
    def _reshape(x):
        # 先把通道维放到第一维，再展平 batch 和空间两维
        x = x.permute(1, 0, 2, 3).contiguous()  # (C, N, H, W)
        return x.view(C, -1)                    # (C, N*H*W)

    Fs_flat = _reshape(Fs)
    Ft_flat = _reshape(Ft)

    # 2) 去中心化
    Fs_mean = Fs_flat.mean(dim=1, keepdim=True)
    Ft_mean = Ft_flat.mean(dim=1, keepdim=True)
    Fs_centered = Fs_flat - Fs_mean
    Ft_centered = Ft_flat - Ft_mean

    # 3) 计算协方差矩阵 (C×C)
    #    注意分母用样本数-1：N*H*W - 1
    num_samples = N * H * W
    cov_s = (Fs_centered @ Fs_centered.t()) / (num_samples - 1)
    cov_t = (Ft_centered @ Ft_centered.t()) / (num_samples - 1)

    # 4) 计算 Frobenius 范数差，并归一化
    #    L_CORAL = (1 / (4 C^2)) * ||cov_s - cov_t||_F^2
    diff = cov_s - cov_t
    loss = torch.sum(diff * diff) / (4 * C * C)

    return loss
