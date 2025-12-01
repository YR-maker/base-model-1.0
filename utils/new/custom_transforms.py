'''

该文件为新增文件，用于在数据加载时生成全局坐标。
这个文件的作用是生成 (3, D, H, W) 的坐标网格，并将其拼接在图像通道之后。
这样，后续的 CenterSpatialCropd 或 RandFlipd 会自动同步处理坐标，保证坐标与解剖位置对应。

'''
import torch
import numpy as np
from monai.transforms import MapTransform
from monai.utils import ensure_tuple_rep


class AddGlobalCoordinatesd(MapTransform):
    """
    自定义变换：添加全局归一化坐标通道。

    该变换会在图像通道维度后追加3个坐标通道 (z, y, x)。
    坐标值会被归一化到 [-1, 1] 之间。

    重要：必须在 CenterSpatialCropd 或 RandomCropd 之前使用，
    这样裁剪后的 patch 才会携带其在原始解剖结构中的绝对位置信息。
    """

    def __init__(self, keys, spatial_dims=3):
        super().__init__(keys)
        self.spatial_dims = spatial_dims

    def __call__(self, data):
        d = dict(data)
        for key in self.keys_tuple:
            # 获取图像 (C, D, H, W)
            img = d[key]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # 获取空间维度 (D, H, W)
            spatial_shape = img.shape[1:]

            # 生成归一化网格 (-1 到 1)
            # indexing='ij' 确保顺序对应 (D, H, W) 即 (z, y, x)
            coords = [torch.linspace(-1.0, 1.0, steps=s) for s in spatial_shape]
            grid = torch.meshgrid(*coords, indexing='ij')

            # 堆叠坐标通道 -> (3, D, H, W)
            coord_map = torch.stack(grid, dim=0).to(img.device, dtype=img.dtype)

            # 将坐标拼接到原始图像后面 -> (C+3, D, H, W)
            d[key] = torch.cat([img, coord_map], dim=0)

            # 标记元数据（可选，用于调试）
            d[f"{key}_meta_dict"] = d.get(f"{key}_meta_dict", {})
            d[f"{key}_meta_dict"]["has_coords"] = True

        return d