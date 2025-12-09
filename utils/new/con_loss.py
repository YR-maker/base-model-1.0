import torch
import torch.nn as nn


class ConnectivityLoss3D(nn.Module):
    """
    3D 连通性损失 (Connectivity Loss)
    参考论文: Masked Vascular Structure Segmentation and Completion

    原理: 通过计算预测图和金标准在 X, Y, Z 三个方向上的梯度(平滑度)差异，
    来惩罚断裂。断裂处的梯度通常会发生剧烈变化。
    """

    def __init__(self, lambda_coeff=1.0):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def smoothness_penalty(self, tensor):
        """
        计算 3D 张量的平滑度惩罚项 (Total Variation 的变体)
        tensor shape: (B, C, D, H, W) 或 (B, D, H, W)
        """
        # 确保是 5 维张量 (Batch, Channel, Depth, Height, Width)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)

        # 计算 X 方向梯度 (Width 维度)
        # tensor[..., 1:] - tensor[..., :-1] 表示相邻像素相减
        grad_x = torch.abs(tensor[..., :, :, 1:] - tensor[..., :, :, :-1])

        # 计算 Y 方向梯度 (Height 维度)
        grad_y = torch.abs(tensor[..., :, 1:, :] - tensor[..., :, :-1, :])

        # 计算 Z 方向梯度 (Depth 维度) -> 3D 特有
        grad_z = torch.abs(tensor[..., 1:, :, :] - tensor[..., :-1, :, :])

        # 求均值 (保持数值稳定性)
        return torch.mean(grad_x) + torch.mean(grad_y) + torch.mean(grad_z)

    def forward(self, pred, target):
        """
        Args:
            pred: 模型的预测输出 logits (未经过 sigmoid)
            target: 真实的二进制标签 (0 或 1)
        """
        # 1. 预处理预测值
        # 论文中使用带温度系数的 Sigmoid 来进行“软二值化”，使其可导
        # 系数 10 可以让 sigmoid 更陡峭，接近 step 函数
        pred_soft = torch.sigmoid(10 * (pred - 0.5))

        # 2. 预处理标签
        # 标签本身是 0/1，但也进行同样的处理以保持一致性，或者直接转 float
        target_soft = target.float()
        # 也可以对 target 做同样的锐化，论文代码中是这样做的：
        # target_soft = torch.sigmoid(10 * (target.float() - 0.5))

        # 3. 计算平滑度 (即“由于断裂造成的梯度突变总量”)
        smooth_pred = self.smoothness_penalty(pred_soft)
        smooth_target = self.smoothness_penalty(target_soft)

        # 4. 归一化因子 (防止除以零)
        # 使用目标血管的体积作为分母，这样损失值不会随血管多少剧烈波动
        normalization = torch.sum(target_soft) + 1e-6

        # 5. 计算损失
        # 目标是让预测图的平滑度特性 逼近 真实图的平滑度特性
        loss = torch.abs(smooth_pred - smooth_target) / normalization

        return self.lambda_coeff * loss