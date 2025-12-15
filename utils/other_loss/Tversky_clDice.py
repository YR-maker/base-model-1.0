import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import TverskyLoss


class SoftSkeletonize(nn.Module):
    """
    软骨架化 (Soft Skeletonization)
    通过迭代的软腐蚀和最大池化，从概率图中提取可导的骨架。
    """

    def __init__(self, num_iter=5):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def forward(self, img):
        # img shape: (B, C, D, H, W)
        p1 = F.max_pool3d(img * -1, (3, 3, 3), 1, 1) * -1
        for _ in range(self.num_iter):
            p1 = F.max_pool3d(p1 * -1, (3, 3, 3), 1, 1) * -1
        return p1


class SoftclDiceLoss(nn.Module):
    """
    Soft-clDice Loss
    专门针对血管的拓扑结构损失，强迫预测结果的骨架与金标准骨架重合。
    """

    def __init__(self, iter_=3, smooth=1.):
        super(SoftclDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true):
        # y_pred: logits (未经过sigmoid)
        # y_true: binary mask

        # 1. 确保概率化 (0-1之间)
        probs = torch.sigmoid(y_pred)

        # 2. 提取预测图的骨架 (Soft Skeleton)
        skel_pred = self.soft_skeletonize(probs)

        # 3. 提取金标准的骨架 (Hard Skeleton converted to Soft form for consistency)
        # 注意：这里直接对 binary target 做同样操作
        skel_true = self.soft_skeletonize(y_true.float())

        # 4. 计算 Tprec (拓扑精确度): 预测的骨架有多少在真实血管里？
        # 使用 probs * skel_true 是为了保持梯度回传
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)

        # 5. 计算 Tsens (拓扑敏感度/召回): 真实的骨架有多少被预测到了？
        tsens = (torch.sum(skel_true * probs) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        # 6. 计算 clDice
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)

        return 1.0 - cl_dice


class HybridVesselLoss(nn.Module):
    """
    【终极组合损失】
    Tversky Loss (负责召回率，防断裂) + Soft-clDice Loss (负责拓扑连接)
    """

    def __init__(self, alpha=0.3, beta=0.7, iter_=3, lambda_cldice=0.5, lambda_main=1.0):
        super(HybridVesselLoss, self).__init__()
        self.lambda_main = lambda_main
        self.lambda_cldice = lambda_cldice

        # 1. Tversky Loss: beta=0.7 表示我们更怕漏检(断裂)，惩罚力度大
        self.main_loss = TverskyLoss(
            include_background=False,
            to_onehot_y=False,
            sigmoid=True,
            alpha=alpha,
            beta=beta
        )

        # 2. clDice Loss
        self.cldice_loss = SoftclDiceLoss(iter_=iter_)

    def forward(self, pred, target):
        loss1 = self.main_loss(pred, target)
        loss2 = self.cldice_loss(pred, target)

        return self.lambda_main * loss1 + self.lambda_cldice * loss2