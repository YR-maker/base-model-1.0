import logging
import torch
from model.module import PLModuleFinetune as BasePLModuleFinetune
from utils.other_loss.skeleton_loss import SkeletonWeightedLoss

logger = logging.getLogger(__name__)


class PLModuleSkeleton(BasePLModuleFinetune):
    """
    【骨架召回微调模块】
    使用 SkeletonWeightedLoss 替代之前的 ConnectivityLoss。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 初始化骨架损失
        # weight=10.0 表示骨架像素的重要性是普通像素的 10 倍
        self.skel_loss_fn = SkeletonWeightedLoss(skeleton_weight=10.0)

        # 损失函数的混合权重
        # 建议 0.5 ~ 1.0。因为 BCE Loss 数值本身较大 (0.1~0.7)，
        # 这里的 lambda 不需要像 TV Loss 那样设很大或很小。
        self.lambda_skel = 1.0

        if self.rank == 0:
            logger.info(f"Initialized PLModuleSkeleton: Weight={self.lambda_skel}, Skel_Emphasis=10.0")

    def training_step(self, batch, batch_idx):
        image, mask = batch

        # 1. 前向传播
        pred_mask = self.model(image)

        # 2. 基础损失 (Dice + CE)
        base_loss = self.loss(pred_mask, mask)

        # 3. 骨架损失 (注意：输入必须是 logits，不要 sigmoid)
        # 你的模型输出通常是 logits，如果 base_loss 里有 sigmoid，这里直接传 pred_mask 即可
        skel_loss = self.skel_loss_fn(pred_mask, mask)

        # 4. 总损失
        total_loss = base_loss + (self.lambda_skel * skel_loss)

        # 5. 日志记录
        if self.rank == 0:
            self.log("train_base_loss", base_loss)
            self.log("train_skel_loss", skel_loss)  # 观察这个数值，应该在 0.1 ~ 1.0 之间
            self.log("train_total_loss", total_loss)

            # 可选：打印调试信息 (仅前几个 step)
            if batch_idx < 5 and self.current_epoch == 0:
                logger.info(f"[Step {batch_idx}] Base: {base_loss:.4f} | Skel: {skel_loss:.4f}")

        return total_loss