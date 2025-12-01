import logging

from monai import transforms
from monai.transforms import Compose

logger = logging.getLogger(__name__)

def generate_transforms(
        transforms_config: list[dict],
) -> list[transforms.Transform]:
    """
    根据配置生成数据变换管道
    训练和推理过程中使用的数据增强和预处理变换。
    Args:
        transforms_config (list[dict]): 变换配置列表，每个字典包含变换名称和参数
    Returns:
        Compose: 组合后的变换管道对象
    """

    transform_list = []  # 存储变换对象的列表
    logger.debug(f"Generating {len(transforms_config)} transforms")  # 记录变换数量

    # 遍历所有变换配置
    for transform_config in transforms_config:
        # 获取变换名称（字典的第一个键）
        transform_name = next(iter(transform_config))
        # 获取变换的参数配置
        transform_kwargs = transform_config[transform_name]
        logger.debug(
            f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        )
        # 使用反射机制动态创建变换对象
        # 从MONAI的transforms模块中获取对应的变换类并实例化
        transform: transforms.Transform = getattr(transforms, transform_name)(
            **transform_kwargs  # 使用关键字参数初始化
        )  # type: ignore
        transform_list.append(transform)  # 将变换添加到列表中

    # 将所有变换组合成一个管道，按顺序执行
    return Compose(transform_list)  # type: ignore