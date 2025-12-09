import logging
import torch
# 继承原来的微调模块
from utils.module import PLModuleFinetune as BasePLModuleFinetune
# 引入刚才定义的损失函数
from utils.new.con_loss import ConnectivityLoss3D

logger = logging.getLogger(__name__)


class PLModuleConnectivityOnly(BasePLModuleFinetune):
    """
    【仅改进损失函数版】
    继承自基础微调模块，仅在 Training Step 中增加了 Connectivity Loss。
    没有任何 Mask 遮罩逻辑。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化连通性损失，权重设为 1.0 (可调)
        self.connect_loss_fn = ConnectivityLoss3D(lambda_coeff=1.0)
        if self.rank == 0:
            logger.info("Initialized PLModuleConnectivityOnly: Added ConnectivityLoss3D")

    def training_step(self, batch, batch_idx):
        image, mask = batch

        # 1. 模型前向传播
        pred_mask = self.model(image)

        # 2. 计算基础损失 (Dice + CE，来自配置文件)
        base_loss = self.loss(pred_mask, mask)

        # 3. 计算额外的连通性损失
        con_loss = self.connect_loss_fn(pred_mask, mask)

        # 4. 总损失相加
        total_loss = base_loss + con_loss

        # 5. 记录日志 (Rank 0)
        if self.rank == 0:
            self.log("train_base_loss", base_loss)
            self.log("train_con_loss", con_loss)  # 重点关注这个指标下降
            self.log("train_loss", total_loss)

        return total_loss