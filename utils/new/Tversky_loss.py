import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, TverskyLoss


class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=5):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        elif len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        else:
            raise ValueError("Input must be 4D or 5D tensor")

    def soft_dilate(self, img):
        if len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        elif len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        else:
            raise ValueError("Input must be 4D or 5D tensor")

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for i in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)


class SoftclDiceLoss(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(SoftclDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true):
        if not ((y_pred >= 0) & (y_pred <= 1)).all():
            y_pred = torch.sigmoid(y_pred)

        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / \
                (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / \
                (torch.sum(skel_true) + self.smooth)

        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


class HybridVesselLoss(nn.Module):
    """
    [升级版] 混合损失：Tversky (防断裂) + clDice (保骨架)
    """

    def __init__(self, alpha=0.3, beta=0.7, lambda_main=1.0, lambda_cldice=0.4, iter_=3):
        super().__init__()
        self.lambda_main = lambda_main
        self.lambda_cldice = lambda_cldice

        # 使用 TverskyLoss 替代 DiceCELoss
        # include_background=True 因为我们通常是单通道二值预测，MONAI处理方式
        # sigmoid=True 确保输入经过 sigmoid
        self.main_loss = TverskyLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            alpha=alpha,  # FP 权重 (噪声)
            beta=beta,  # FN 权重 (断裂/漏标) -> 设大一点！
        )
        self.cldice_loss = SoftclDiceLoss(iter_=iter_)

    def forward(self, y_pred, y_true):
        # 1. 主损失 (Tversky)
        loss_main = self.main_loss(y_pred, y_true)

        # 2. 拓扑损失 (clDice)
        y_pred_prob = torch.sigmoid(y_pred)
        loss_cl = self.cldice_loss(y_pred_prob, y_true)

        # 3. 组合
        total_loss = (self.lambda_main * loss_main) + (self.lambda_cldice * loss_cl)
        return total_loss