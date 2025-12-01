"""Reader/writer classes; we follow https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/imageio"""
"""
医学图像读写器类 - 支持多种医学图像格式

基于nnUNet的图像IO模块设计，为vesselFM提供统一的数据接口
支持论文中提到的多种成像模态数据格式
"""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class BaseReaderWriter(ABC):
    """基础读写器抽象基类

    定义统一的医学图像读写接口，确保不同格式数据的一致性处理
    对应论文中处理多种成像模态数据的需求
    """
    supported_file_formats = []  # 支持的文件格式列表

    @staticmethod
    def _check_all_same(lst: List) -> bool:
        """
        检查列表中所有元素是否相同

        Args:
            lst: 元素列表

        Returns:
            bool: 所有元素是否相同
        """
        return all(x == lst[0] for x in lst)

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        从磁盘读取图像数据
        Args:
            image_fnames: 图像文件名列表
        Returns:
            Tuple[np.ndarray, dict]: 图像数组和元数据字典
        """
        pass

    @abstractmethod
    def read_segs(self, seg_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        从磁盘读取分割标签数据
        Args:
            seg_fnames: 分割标签文件名列表
        Returns:
            Tuple[np.ndarray, dict]: 分割标签数组和元数据字典
        """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, seg_fname: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        将分割结果写入磁盘
        Args:
            seg: 分割结果数组
            seg_fname: 输出文件名
            metadata: 元数据字典（包含空间信息等）
        """
        pass
class NumpyReaderWriter(BaseReaderWriter):
    """Numpy格式读写器

    专门处理.npy和.npz格式的3D图像数据
    适用于预处理后的数组数据存储和读取
    """
    supported_file_formats = ["npy", "npz"]  # 支持的numpy格式

    def __init__(self):
        super().__init__()

    def read_images(
        self, image_fnames: Union[str, list[str]], metdata_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取numpy格式的图像数据

        Args:
            image_fnames: 图像文件名或列表
            metdata_path: 元数据文件路径（可选）

        Returns:
            Tuple[np.ndarray, Dict]: 图像数据数组和元数据字典
        """
        image_data = []
        if type(image_fnames) is str:
            image_fnames = [image_fnames]  # 统一转换为列表处理

        for image_fname in image_fnames:
            # 提取文件扩展名并验证支持性
            file_extension = os.path.basename(image_fname).split(".")[-1]
            print(file_extension)
            if file_extension not in self.supported_file_formats:
                raise RuntimeError(f"File format not supported for {image_fname}")

            if file_extension == "npy":
                # 读取单个npy文件
                image_data.append(self._load_npy(image_fname))
                if image_data[-1].ndim != 3:
                    raise RuntimeError(
                        f"Image {image_fname} has dimension {image_data[-1].ndim}, expected 3"
                    )
            elif file_extension == "npz":
                # 读取npz压缩文件（可能包含多个数组）
                new_images = self._load_npz(image_fname)
                for image in new_images:
                    if image.ndim != 3:
                        raise RuntimeError(
                            f"Image in {image_fname} has dimension {image.ndim}, expected 3"
                        )
                image_data.extend(new_images)

        # 合并多个图像数据
        if len(image_data) > 1:
            image_data = np.vstack(image_data)
        else:
            image_data = image_data[0]

        # 处理元数据
        if metdata_path is not None:
            with open(metdata_path, "r") as f:
                metadata = json.load(f)
            spacing = metadata["spacing"]
            return image_data, {"spacing": spacing, "other": metadata}
        else:
            spacing = [1, 1, 1]  # 默认间距
            return image_data, {"spacing": spacing}

    def _load_npy(self, fname: str) -> np.ndarray:
        """加载单个npy文件"""
        return np.load(fname)

    def _load_npz(self, fname: str) -> list[np.ndarray]:
        """加载npz文件中的所有数组"""
        file = np.load(fname)
        array = []
        for key in file.files:  # 遍历所有压缩的数组
            array.append(file[key])
        return array

    def read_segs(
        self, seg_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取分割标签数据（复用图像读取逻辑）"""
        return self.read_images(seg_fnames)

    def write_seg(
        self,
        seg: np.ndarray,
        seg_fname: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """写入分割结果到numpy格式"""
        file_extension = os.path.basename(seg_fname).split(".")[-1]
        if file_extension != ".npy":
            raise RuntimeError(
                f"File format {file_extension} not supported,  saving {seg_fname} failed!"
            )
        if metadata is not None:
            spacing = metadata["spacing"]
            with open(seg_fname, "wb") as f:
                np.save(f, seg)
            # 同时保存元数据到JSON文件
            with open(seg_fname.replace(".npy", ".json"), "w") as f:
                json.dump(metadata, f)
        else:
            with open(seg_fname, "wb") as f:
                np.save(f, seg)

class NumpySeriesReaderWriter(BaseReaderWriter):
    """Numpy序列读写器

    专门处理2D切片序列数据，适用于切片形式的医学图像
    支持按索引范围读取部分切片
    """
    supported_file_formats = ["npy_series", "npz_series"]  # 序列格式标识
    slice_formats = ["npy", "npz"]  # 切片文件格式

    def __init__(self):
        super().__init__()

    def read_images(
        self,
        image_folder: str,
        metdata_path: Optional[str] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        """
        读取切片序列图像数据

        Args:
            image_folder: 包含切片的文件夹路径
            metdata_path: 元数据文件路径
            start_idx: 起始切片索引
            end_idx: 结束切片索引

        Returns:
            Tuple[np.ndarray, dict]: 堆叠后的3D数组和元数据
        """
        # 获取文件夹中所有切片文件
        image_files = os.listdir(image_folder)
        image_files = [
            f for f in image_files if f.endswith(".npy") or f.endswith(".npz")
        ]

        # 按索引范围过滤文件
        if start_idx is not None:
            image_files = [f for f in image_files if int(f.split(".")[0]) >= start_idx]
        if end_idx is not None:
            image_files = [f for f in image_files if int(f.split(".")[0]) < end_idx]

        image_files.sort()  # 按文件名排序
        image_files = [os.path.join(image_folder, f) for f in image_files]

        image_data = []
        for image_fname in image_files:
            file_extension = os.path.basename(image_fname).split(".")[-1].lower()
            if file_extension not in self.slice_formats:
                raise RuntimeError(f"File format not supported for {image_fname}")

            if file_extension == "npy":
                image = self._load_npy(image_fname)
                image_data.append(image)
                if image_data[-1].ndim != 2:
                    raise RuntimeError(
                        f"Image {image_fname} has dimension {image_data[-1].ndim}, expected 2"
                    )
            elif file_extension == "npz":
                new_images = self._load_npz(image_fname)
                for image in new_images:
                    if image.ndim != 2:
                        raise RuntimeError(
                            f"Image in {image_fname} has dimension {image.ndim}, expected 2"
                        )
                image_data.extend(new_images)

        # 堆叠2D切片形成3D体积
        if metdata_path is not None:
            with open(metdata_path, "r") as f:
                metadata = json.load(f)
            return np.stack(image_data, axis=0), metadata
        else:
            spacing = [1, 1, 1]
            return np.stack(image_data, axis=0), {"spacing": spacing}

    def _load_npy(self, fname: str) -> np.ndarray:
        """加载单个npy切片文件"""
        return np.load(fname)

    def _load_npz(self, fname: str) -> list[np.ndarray]:
        """加载npz文件中的所有切片"""
        file = np.load(fname)
        array = []
        for key in file.files:
            array.append(file[key])
        return array

    def read_segs(
        self,
        seg_fnames: Union[str, list[str]],
        metadata: Optional[Dict[str, Any]] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取分割标签序列数据"""
        return self.read_images(seg_fnames, metadata, start_idx, end_idx)

    def _save_npy_series(self, array: np.ndarray, output_folder: str):
        """保存为numpy切片序列"""
        os.makedirs(output_folder, exist_ok=True)
        for i in range(array.shape[0]):
            # 按切片索引保存为单独文件
            np.save(os.path.join(output_folder, f"{i}.npy"), array[i])

    def write_seg(
        self,
        seg: np.ndarray,
        seg_fname: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """写入分割结果序列"""
        if metadata is not None:
            self._save_npy_series(seg, seg_fname)
            json.dump(metadata, os.path.join(seg_fname, "metadata.json"))
        else:
            self._save_npy_series(seg, seg_fname)


class SimpleITKReaderWriter(BaseReaderWriter):
    """SimpleITK格式读写器

    支持标准医学图像格式：NIfTI, MHA, MHD, NRRD等
    保留完整的医学图像元数据（间距、原点、方向等）
    """
    supported_file_formats = ["nii", "nii.gz", "mha", "mhd", "nrrd", "gz"]

    def __init__(self):
        super().__init__()

    def read_images(
        self, image_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取SimpleITK支持的医学图像格式

        Args:
            image_fnames: 图像文件名列表

        Returns:
            Tuple[np.ndarray, Dict]: 图像数据和完整的元数据
        """

        if type(image_fnames) is not list:
            image_fnames = [image_fnames]

        image_data = []
        image_metadata = {"spacing": [], "origin": [], "direction": []}

        for image_fname in image_fnames:
            # 验证文件格式支持性
            if (
                os.path.basename(image_fname).split(".")[-1]
                not in self.supported_file_formats
            ):
                raise RuntimeError(f"File format not supported for {image_fname}")

            # 使用SimpleITK读取图像
            image = sitk.ReadImage(image_fname)
            logger.debug(f"Image {image_fname} has shape {image.GetSize()}")
            image_data.append(sitk.GetArrayFromImage(image))

            if image_data[-1].ndim != 3:
                raise RuntimeError(
                    f"Image {image_fname} has dimension {image_data[-1].ndim}, expected 3"
                )

            # 收集元数据信息
            image_metadata["spacing"].append(image.GetSpacing())
            image_metadata["origin"].append(image.GetOrigin())
            image_metadata["direction"].append(image.GetDirection())

        # 验证多个图像的元数据一致性
        if not self._check_all_same(image_metadata["spacing"]):
            logger.error("Spacing is not the same for all images")
            raise RuntimeError("Spacing is not the same for all images")
        if not self._check_all_same(image_metadata["origin"]):
            logger.warning("Origin is not the same for all images")
            logger.warning("Please check if this is expected behavior")
        if not self._check_all_same(image_metadata["direction"]):
            logger.warning("Direction is not the same for all images")
            logger.warning("Please check if this is expected behavior")

        # 提取统一的元数据
        sitk_metadata = {}
        for key in image_metadata.keys():
            sitk_metadata[key] = image_metadata[key][0]

        # 调整间距顺序以匹配numpy数组维度顺序 (z, y, x) -> (z, x, y)
        spacing = [
            sitk_metadata["spacing"][2],  # z-spacing
            sitk_metadata["spacing"][0],  # x-spacing
            sitk_metadata["spacing"][1],  # y-spacing
        ]
        meta_data = {"spacing": spacing, "other": sitk_metadata}
        logger.debug(f"Spacing: {spacing}")
        logger.debug(f"Final shape: {np.vstack(image_data).shape}")
        return np.vstack(image_data), meta_data

    def read_segs(
        self, seg_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取分割标签数据"""
        return self.read_images(seg_fnames)

    def write_seg(
        self,
        seg: np.ndarray,
        seg_fname: str,
        metadata: Optional[Dict[str, Any]] = None,
        compression: bool = True,
    ):
        """写入分割结果到医学图像格式"""
        seg = sitk.GetImageFromArray(seg)
        if metadata is not None:
            # 恢复元数据信息
            seg.SetSpacing(metadata["spacing"])
            seg.SetOrigin(metadata["origin"])
            seg.SetDirection(metadata["direction"])

        sitk.WriteImage(seg, seg_fname, compression)


def determine_reader_writer(file_ending: str):
    """
    根据文件扩展名自动确定合适的读写器------------------------------------------------------------------------
    Args:
        file_ending: 文件扩展名
    Returns:
        BaseReaderWriter: 对应的读写器类
    """
    LIST_OF_READERS_WRITERS = [
        SimpleITKReaderWriter,
        NumpyReaderWriter,
        NumpySeriesReaderWriter,
    ]

    for reader_writer in LIST_OF_READERS_WRITERS:
        if file_ending.lower() in reader_writer.supported_file_formats:
            logger.debug(
                f"Automatically determined reader_writer: {reader_writer.__name__} for file ending: {file_ending}"
            )
            return reader_writer

    raise ValueError(f"No reader_writer found for file ending: {file_ending}")