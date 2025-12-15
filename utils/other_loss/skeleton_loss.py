import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonWeightedLoss(nn.Module):
    """
    【骨架加权损失 (Skeleton-Weighted Loss)】

    专为少样本血管分割设计。
    原理：
    1. 实时计算 Ground Truth 的骨架 (通过形态学腐蚀)。
    2. 生成权重图：背景和普通血管权重为 1.0，但骨架区域权重极高 (如 10.0)。
    3. 计算加权 BCE Loss。

    优势：
    强迫模型优先拟合血管的“连通核心”，避免因追求平滑而擦除细小血管。
    """

    def __init__(self, skeleton_weight=10.0):
        """
        Args:
            skeleton_weight: 骨架区域的权重倍数。建议 10.0 ~ 20.0。
                             数值越大，模型越不敢断开血管。
        """
        super().__init__()
        self.skeleton_weight = skeleton_weight
        # 3D MaxPool 用于模拟形态学腐蚀 (Erosion)
        # kernel=3, stride=1, padding=1 保持尺寸不变
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

    def get_approx_skeleton(self, mask):
        """
        提取近似骨架。
        逻辑：骨架 = 原始掩码 - 腐蚀后的掩码
        """
        # 确保 mask 是 float (0.0 或 1.0)
        mask = mask.float()

        # 模拟腐蚀: -MaxPool(-mask) 等价于 MinPool
        # 这一步会把血管剥去一层皮
        eroded = -self.max_pool(-mask)

        # 原始减去腐蚀，剩下的就是刚才被剥掉的“皮”和极细血管的“骨”
        # 对于细血管，腐蚀后消失，相减即为本身
        skeleton = mask - eroded
        return skeleton

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: 模型的原始输出 (未经过 sigmoid)
            target: Ground Truth (0/1)
        """
        # 1. 确保 target 类型正确
        target = target.float()

        # 2. 计算基础的 BCE Loss (不求平均，保留每个像素的 Loss)
        # output: [Batch, Channel, D, H, W]
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        # 3. 动态提取 Ground Truth 的骨架
        with torch.no_grad():
            # 计算骨架 (不需要梯度)
            skel_map = self.get_approx_skeleton(target)

            # 4. 构建权重图
            # 基础权重 = 1.0
            # 骨架区域 = 1.0 + skeleton_weight (例如 11.0)
            weight_map = 1.0 + (skel_map > 0.5).float() * self.skeleton_weight

        # 5. 应用权重并求平均
        weighted_loss = bce_loss * weight_map

        return weighted_loss.mean()