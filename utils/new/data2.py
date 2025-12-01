
'''

该文件为新增文件，原本的 data.py 只能加载 MONAI 自带的变换。
我们需要修改 generate_transforms 函数，让它也能识别我们刚刚写的 AddGlobalCoordinatesd。

'''

import logging
from monai import transforms
from monai.transforms import Compose

# === 【新增】导入我们自定义的变换模块 ===
import utils.new.custom_transforms as custom_transforms

# ======================================

logger = logging.getLogger(__name__)


def generate_transforms(
        transforms_config: list[dict],
) -> list[transforms.Transform]:
    """
    根据配置生成数据变换管道
    """
    transform_list = []
    logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        logger.debug(
            f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        )

        # === 【修改核心逻辑】 ===
        # 先尝试从 MONAI 标准库中加载
        if hasattr(transforms, transform_name):
            transform_cls = getattr(transforms, transform_name)
        # 如果没有，尝试从我们的自定义模块加载
        elif hasattr(custom_transforms, transform_name):
            transform_cls = getattr(custom_transforms, transform_name)
        else:
            raise ImportError(f"无法找到变换类: {transform_name}，请检查是否拼写错误或未导入自定义模块。")

        transform = transform_cls(**transform_kwargs)
        # ======================

        transform_list.append(transform)

    return Compose(transform_list)