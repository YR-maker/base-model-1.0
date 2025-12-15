import torch
import torch.nn as nn


class ConnectivityLoss3D(nn.Module):
    """
    3D 连通性损失 (Connectivity Loss) - 修正版
    """

    def __init__(self, lambda_coeff=1.0): # 建议系数保持 1.0 或 5.0 即可
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def smoothness_penalty(self, tensor):
        """
        计算 3D 张量的平滑度惩罚项 (Total Variation)
        【修正】这里必须用 Sum 而不是 Mean，否则数值会过小
        """
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)

        # 计算 X, Y, Z 三个方向的梯度绝对值
        grad_x = torch.abs(tensor[..., :, :, 1:] - tensor[..., :, :, :-1])
        grad_y = torch.abs(tensor[..., :, 1:, :] - tensor[..., :, :-1, :])
        grad_z = torch.abs(tensor[..., 1:, :, :] - tensor[..., :-1, :, :])

        # 【核心修改】将 torch.mean 改为 torch.sum
        return torch.sum(grad_x) + torch.sum(grad_y) + torch.sum(grad_z)

    def forward(self, pred, target):
        """
        Args:
            pred: 模型的预测输出 logits
            target: 真实的二进制标签
        """
        # 1. 预处理
        pred_soft = torch.sigmoid(10 * (pred - 0.5))
        target_soft = target.float()

        # 2. 计算平滑度 (现在是 Total Variation，即总表面积概念)
        smooth_pred = self.smoothness_penalty(pred_soft)
        smooth_target = self.smoothness_penalty(target_soft)

        # 3. 归一化因子 (血管体积)
        # 加上 1e-4 防止除零
        normalization = torch.sum(target_soft) + 1e-4

        # 4. 计算损失
        # (预测的表面积/体积 - 真实的表面积/体积)
        # 这个差值通常在 0.01 ~ 0.5 之间
        loss = torch.abs(smooth_pred - smooth_target) / normalization

        return self.lambda_coeff * loss