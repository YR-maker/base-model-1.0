import nibabel as nib
import numpy as np
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import scipy.ndimage as ndimage


class LungROISegmentor:
    """
    肺部ROI分割处理器
    实现肺部区域分割并创建遮罩
    """

    def __init__(self, window_min=-1000, window_max=400, lung_threshold=-400, min_volume=1000):
        """
        初始化肺部ROI分割参数

        Args:
            window_min (int): 预处理窗口下限(HU)，默认-1000
            window_max (int): 预处理窗口上限(HU)，默认400
            lung_threshold (int): 肺部组织阈值(HU)，默认-400
            min_volume (int): 最小肺部区域体积(体素数)，默认1000
        """
        self.window_min = window_min
        self.window_max = window_max
        self.lung_threshold = lung_threshold
        self.min_volume = min_volume

        # 配置日志系统
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('lung_roi_segmentation.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"初始化肺部ROI分割器: 窗口[{window_min}, {window_max}]HU, 肺部阈值{lung_threshold}HU")

    def validate_parameters(self):
        """
        验证参数合理性
        """
        if self.window_min >= self.window_max:
            raise ValueError(f"窗宽下限({self.window_min})必须小于上限({self.window_max})")

        if self.lung_threshold >= self.window_max:
            raise ValueError(f"肺部阈值({self.lung_threshold})应小于窗口上限({self.window_max})")

        if self.lung_threshold <= self.window_min:
            self.logger.warning(f"肺部阈值{self.lung_threshold}HU接近窗口下限{self.window_min}HU")

    def preprocess_ct_data(self, ct_data):
        """
        预处理CT数据：应用窗口和标准化

        Args:
            ct_data (numpy.ndarray): 原始CT数据

        Returns:
            numpy.ndarray: 预处理后的数据
        """
        self.logger.info("开始CT数据预处理")

        # 记录原始数据统计
        original_stats = {
            'min_hu': float(np.min(ct_data)),
            'max_hu': float(np.max(ct_data)),
            'mean_hu': float(np.mean(ct_data)),
            'std_hu': float(np.std(ct_data)),
            'total_voxels': ct_data.size
        }

        self.logger.info(f"原始HU统计: 范围[{original_stats['min_hu']:.1f}, {original_stats['max_hu']:.1f}], "
                         f"均值={original_stats['mean_hu']:.1f}±{original_stats['std_hu']:.1f}")

        # 应用窗口优化
        windowed_data = np.clip(ct_data, self.window_min, self.window_max)

        # 标准化到[0, 1]范围
        normalized_data = (windowed_data - self.window_min) / (self.window_max - self.window_min)
        normalized_data = np.clip(normalized_data, 0, 1)

        return normalized_data, original_stats

    def create_lung_mask(self, ct_data):
        """
        创建肺部区域遮罩

        Args:
            ct_data (numpy.ndarray): 原始CT数据(HU值)

        Returns:
            numpy.ndarray: 肺部遮罩(二值图像)
            dict: 分割统计信息
        """
        self.logger.info("开始肺部区域分割")

        # 初步阈值分割：选择肺部组织(HU值较低的区域)
        initial_mask = ct_data < self.lung_threshold

        # 形态学操作去除小噪声
        structure = ndimage.generate_binary_structure(3, 2)
        cleaned_mask = ndimage.binary_opening(initial_mask, structure=structure)
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=structure)

        # 连通组件分析，选择主要肺部区域
        labeled_mask, num_features = ndimage.label(cleaned_mask)
        region_sizes = np.bincount(labeled_mask.ravel())

        # 选择体积足够大的区域（排除小噪声）
        large_regions = []
        lung_volume = 0

        for i in range(1, num_features + 1):
            if region_sizes[i] >= self.min_volume:
                large_regions.append(i)
                lung_volume += region_sizes[i]

        # 创建最终肺部遮罩
        lung_mask = np.isin(labeled_mask, large_regions)

        # 填充肺部区域内部的空洞
        lung_mask = ndimage.binary_fill_holes(lung_mask)

        # 统计信息
        segmentation_stats = {
            'total_voxels': ct_data.size,
            'lung_voxels': int(np.sum(lung_mask)),
            'lung_volume_ratio': float(np.sum(lung_mask) / ct_data.size),
            'num_lung_regions': len(large_regions),
            'lung_threshold_used': self.lung_threshold
        }

        self.logger.info(f"肺部分割完成: 找到{segmentation_stats['num_lung_regions']}个肺部区域, "
                         f"体积占比{segmentation_stats['lung_volume_ratio']:.2%}")

        return lung_mask.astype(np.uint8), segmentation_stats

    def apply_lung_roi(self, ct_data, lung_mask):
        """
        应用肺部ROI：只保留肺部区域，其他区域设为遮罩值

        Args:
            ct_data (numpy.ndarray): 原始CT数据
            lung_mask (numpy.ndarray): 肺部遮罩

        Returns:
            numpy.ndarray: ROI处理后的数据
        """
        self.logger.info("应用肺部ROI遮罩")

        # 创建ROI图像：肺部区域保留原值，其他区域设为最小值
        roi_data = ct_data.copy()

        # 使用窗口下限作为背景值（通常是最小HU值）
        background_value = self.window_min

        # 应用遮罩：非肺部区域设为背景值
        roi_data[lung_mask == 0] = background_value

        # 统计ROI效果
        preserved_voxels = np.sum(lung_mask)
        masked_voxels = ct_data.size - preserved_voxels
        mask_ratio = masked_voxels / ct_data.size

        self.logger.info(f"ROI应用完成: 保留{preserved_voxels}个体素, "
                         f"遮罩{masked_voxels}个体素({mask_ratio:.2%})")

        return roi_data

    def process_single_image(self, input_path, output_path):
        """
        处理单个图像文件：分割肺部ROI

        Args:
            input_path (str): 输入文件路径
            output_path (str): 输出文件路径

        Returns:
            dict: 处理结果统计
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"处理图像: {input_path}")

            # 验证参数
            self.validate_parameters()

            # 加载NIfTI图像
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"输入文件不存在: {input_path}")

            ct_image = nib.load(input_path)
            ct_data = ct_image.get_fdata()

            self.logger.info(f"图像加载成功: 尺寸{ct_data.shape}, 数据类型{ct_data.dtype}")

            # 预处理CT数据
            normalized_data, original_stats = self.preprocess_ct_data(ct_data)

            # 创建肺部遮罩（使用原始HU值进行分割）
            lung_mask, segmentation_stats = self.create_lung_mask(ct_data)

            # 应用肺部ROI
            roi_data = self.apply_lung_roi(ct_data, lung_mask)

            # 创建新的NIfTI图像
            roi_image = nib.Nifti1Image(
                roi_data.astype(np.float32),
                ct_image.affine,
                ct_image.header
            )

            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # 保存ROI图像
            nib.save(roi_image, output_path)

            # 可选：保存肺部遮罩用于可视化验证
            mask_output_path = str(output_path).replace('.nii.gz', '_mask.nii.gz')
            mask_image = nib.Nifti1Image(
                lung_mask.astype(np.uint8) * 255,  # 转换为0-255范围便于可视化
                ct_image.affine,
                ct_image.header
            )
            nib.save(mask_image, mask_output_path)

            processing_time = (datetime.now() - start_time).total_seconds()

            # 汇总处理结果
            result = {
                'status': 'success',
                'input_path': input_path,
                'output_path': output_path,
                'mask_path': mask_output_path,
                'processing_time': processing_time,
                'original_stats': original_stats,
                'segmentation_stats': segmentation_stats,
                'image_shape': ct_data.shape
            }

            self.logger.info(f"处理完成: {output_path} (耗时: {processing_time:.2f}秒)")

            return result

        except Exception as e:
            error_result = {
                'status': 'error',
                'input_path': input_path,
                'error_message': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            self.logger.error(f"处理失败 {input_path}: {str(e)}")
            return error_result


def find_nifti_files(input_dir):
    """
    在输入目录中查找所有NIfTI文件

    Args:
        input_dir (str): 输入目录路径

    Returns:
        list: NIfTI文件路径列表
    """
    input_path = Path(input_dir)

    # 支持多种NIfTI格式
    nifti_extensions = ['*.nii', '*.nii.gz', '*.hdr', '*.img']
    nifti_files = []

    for ext in nifti_extensions:
        nifti_files.extend(input_path.rglob(ext))

    # 去重和排序
    nifti_files = sorted(list(set(nifti_files)))

    return nifti_files


def batch_process(input_dir, output_dir, window_min=-1000, window_max=400,
                  lung_threshold=-400, min_volume=1000):
    """
    批量处理文件夹中的所有NIfTI图像

    Args:
        input_dir (str): 输入文件夹路径
        output_dir (str): 输出文件夹路径
        window_min (int): 窗口下限(HU)
        window_max (int): 窗口上限(HU)
        lung_threshold (int): 肺部组织阈值(HU)
        min_volume (int): 最小肺部区域体积
    """
    logger = logging.getLogger(__name__)

    # 创建分割器实例
    segmentor = LungROISegmentor(window_min, window_max, lung_threshold, min_volume)

    # 查找所有NIfTI文件
    nifti_files = find_nifti_files(input_dir)

    if not nifti_files:
        logger.error(f"在目录 {input_dir} 中未找到NIfTI文件")
        return

    logger.info(f"找到 {len(nifti_files)} 个NIfTI文件")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    total_lung_ratio = 0.0

    # 处理每个文件
    for input_file in tqdm(nifti_files, desc="肺部ROI分割进度"):
        try:
            # 生成输出文件名：原文件名_LungROI.nii.gz
            stem = input_file.stem
            if input_file.suffix == '.gz':  # 处理.nii.gz情况
                stem = input_file.stem.split('.')[0]

            output_filename = f"{stem}_LungROI.nii.gz"
            output_path = Path(output_dir) / output_filename

            # 处理图像
            result = segmentor.process_single_image(str(input_file), str(output_path))

            if result['status'] == 'success':
                successful += 1
                total_lung_ratio += result['segmentation_stats']['lung_volume_ratio']

                # 记录详细统计
                logger.debug(f"成功处理: {input_file.name} -> "
                             f"肺部体积占比: {result['segmentation_stats']['lung_volume_ratio']:.2%}")
            else:
                failed += 1

        except Exception as e:
            logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
            failed += 1

    # 输出批量处理统计
    logger.info("\n" + "=" * 70)
    logger.info("肺部ROI分割批量处理统计报告")
    logger.info("=" * 70)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"处理参数: 窗口[{window_min}, {window_max}]HU, 肺部阈值{lung_threshold}HU")
    logger.info(f"处理文件总数: {len(nifti_files)}")
    logger.info(f"成功处理: {successful} 个文件")
    logger.info(f"处理失败: {failed} 个文件")

    if successful > 0:
        avg_lung_ratio = total_lung_ratio / successful
        logger.info(f"平均肺部体积占比: {avg_lung_ratio:.2%}")

    logger.info("=" * 70)


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(
        description='肺部ROI分割程序 - 提取肺部区域并添加遮罩',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
参数说明:
  - 肺部阈值(lung_threshold): 通常-400到-600 HU，值越小包含的肺部组织越多
  - 最小体积(min_volume): 过滤小噪声区域，根据图像分辨率调整
  - 建议先使用默认参数，然后根据结果微调

输出文件:
  - 原文件名_LungROI.nii.gz: 肺部ROI图像（非肺部区域被遮罩）
  - 原文件名_LungROI_mask.nii.gz: 肺部遮罩图像（用于可视化验证）
        """
    )

    parser.add_argument('--input', '-i', default="/home/yangrui/Project/Base-models/input/Parse/img",
                        help='输入目录路径，包含NIfTI格式的CT图像')
    parser.add_argument('--output', '-o', default="/home/yangrui/Project/Base-models/input/Parse/ROI",
                        help='输出目录路径')
    parser.add_argument('--window_min', type=int, default=-1000,
                        help='预处理窗口下限(HU)，默认-1000')
    parser.add_argument('--window_max', type=int, default=0,
                        help='预处理窗口上限(HU)，默认400')
    parser.add_argument('--lung_threshold', type=int, default=-400,
                        help='肺部组织阈值(HU)，低于此值被认为是肺部组织，默认-400')
    parser.add_argument('--min_volume', type=int, default=1000,
                        help='最小肺部区域体积(体素数)，用于过滤噪声，默认700')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细日志模式')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return

    print("=" * 70)
    print("肺部ROI分割程序")
    print("=" * 70)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"处理参数: 窗口[{args.window_min}, {args.window_max}]HU")
    print(f"肺部阈值: {args.lung_threshold}HU, 最小体积: {args.min_volume}体素")
    print("=" * 70)

    try:
        # 执行批量处理


        batch_process(args.input, args.output, args.window_min, args.window_max,
                      args.lung_threshold, args.min_volume)
        print("\n处理完成! 查看详细日志请检查 lung_roi_segmentation.log 文件")

    except Exception as e:
        print(f"程序执行错误: {str(e)}")
        logging.error(f"程序异常: {str(e)}")


if __name__ == "__main__":
    main()