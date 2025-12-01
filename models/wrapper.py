import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet
import logging

logger = logging.getLogger(__name__)


class SpatialAdapter(nn.Module):
    """
    空间适配器模块
    将 3通道坐标图 转换为 空间门控信号 (Spatial Gate)
    """

    def __init__(self, feature_channels, bottleneck_dim=32):
        super().__init__()
        # 坐标编码器：3 -> 32 -> 32
        self.encoder = nn.Sequential(
            nn.Conv3d(3, bottleneck_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(bottleneck_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(bottleneck_dim),
            nn.LeakyReLU(inplace=True)
        )

        # 映射器：32 -> 特征通道数 (例如 320)
        self.mapper = nn.Conv3d(bottleneck_dim, feature_channels, kernel_size=1)

        # 初始化为 0，确保训练开始时 Adapter 不干扰原模型
        nn.init.zeros_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

    def forward(self, coords, target_shape):
        """
        coords: (B, 3, D_in, H_in, W_in) 原始输入分辨率的坐标
        target_shape: (B, C, D_feat, H_feat, W_feat) 目标特征图的形状
        """
        # 1. 将坐标图下采样到特征图的尺寸
        # align_corners=False 对应于标准的坐标变换
        resized_coords = F.interpolate(
            coords,
            size=target_shape[2:],
            mode='trilinear',
            align_corners=False
        )

        # 2. 提取坐标特征
        feat = self.encoder(resized_coords)

        # 3. 映射并生成门控系数 (0~1)
        gate = torch.sigmoid(self.mapper(feat))
        return gate


class AnatomyAwareVesselFM(nn.Module):
    """
    包装类：冻结的 vesselFM 骨干 + 可训练的 Spatial Adapter
    """

    def __init__(self, base_model_config, pretrained_path=None):
        super().__init__()

        # 1. 实例化基础模型 (DynUNet)
        # 注意：这里我们用 base_model_config 来初始化 DynUNet
        self.backbone = DynUNet(**base_model_config)

        # 2. 加载预训练权重 (如果有)
        if pretrained_path:
            self._load_frozen_weights(pretrained_path)
        else:
            logger.warning("未提供预训练路径，backbone 将随机初始化！")

        # 3. 冻结骨干网络 (Freeze Backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Base vesselFM backbone frozen.")

        # 4. 初始化适配器
        # 获取 bottleneck 的通道数 (根据 dyn_unet_base.yaml 中的 filters 最后一个值: 320)
        bottleneck_channels = base_model_config['filters'][-1]
        self.adapter = SpatialAdapter(feature_channels=bottleneck_channels)

        # 5. 用于存储 hook 的句柄
        self.hook_handle = None

    def _load_frozen_weights(self, path):
        """加载权重并处理可能的 key 不匹配问题"""
        try:
            # 映射到 CPU 加载，避免 OOM
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # 兼容旧版 torch
            checkpoint = torch.load(path, map_location="cpu")

        # 提取 state_dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("models", checkpoint))
        else:
            state_dict = checkpoint

        # 移除 'models.' 前缀 (因为之前的 LightningModule 可能加了前缀)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("models.", "") if k.startswith("models.") else k
            new_state_dict[new_key] = v

        # 加载权重，允许非严格匹配 (以防万一)
        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            logger.warning(f"Backbone loading missing keys: {missing}")
        logger.info(f"Successfully loaded frozen weights from {path}")

    def forward(self, x):
        """
        x: (Batch, 4, D, H, W) -> [Image(1) + Coords(3)]
        """
        # 1. 拆分输入数据
        # Image: Channel 0
        img = x[:, 0:1, ...]
        # Coords: Channel 1, 2, 3
        coords = x[:, 1:, ...]

        # 2. 定义 Hook 函数 (闭包)
        # 这个函数会在 DynUNet 的 bottleneck 层输出时被调用
        def bottleneck_hook(module, input, output):
            # output 是 bottleneck 的特征图
            # 计算适配器的门控信号
            gate = self.adapter(coords, output.shape)
            # 应用门控：抑制非目标区域的特征
            # Skip Connection 之前的特征会被抑制
            return output * gate

        # 3. 注册 Hook 到 DynUNet 的 bottleneck 层
        # DynUNet 的 bottleneck 属性名就是 'bottleneck'
        self.hook_handle = self.backbone.bottleneck.register_forward_hook(bottleneck_hook)

        try:
            # 4. 前向传播 (img 进入骨干网络)
            logits = self.backbone(img)
        finally:
            # 5. 必须移除 Hook，否则下次 forward 会重复注册或报错
            if self.hook_handle:
                self.hook_handle.remove()

        return logits