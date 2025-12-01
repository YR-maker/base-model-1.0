import numpy as np
import nibabel as nib
from scipy import ndimage
import os
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


class CardioPulmonaryExtractor:
    """
    心肺联合区域提取器
    先提取肺部区域，再提取心脏区域，最后合并
    """

    def __init__(self, lung_window_min=-1000, lung_window_max=-400,
                 heart_window_min=-180, heart_window_max=None):
        """
        初始化参数

        Args:
            lung_window_min: 肺部提取阈值下限
            lung_window_max: 肺部提取阈值上限
            heart_window_min: 心脏提取阈值下限
            heart_window_max: 心脏提取阈值上限
        """
        self.lung_window_min = lung_window_min
        self.lung_window_max = lung_window_max
        self.heart_window_min = heart_window_min
        self.heart_window_max = heart_window_max

        print("初始化心肺联合提取器...")
        print(f"肺部提取窗口: [{lung_window_min}, {lung_window_max}] HU")
        print(f"心脏提取窗口: [{heart_window_min}, {heart_window_max or 'auto'}] HU")

    def extract_lungs(self, data):
        """
        提取肺部区域（左右肺）
        """
        print("步骤1: 提取肺部区域...")

        # 创建肺部二值掩码
        lung_mask = np.logical_and(data >= self.lung_window_min,
                                   data <= self.lung_window_max)
        lung_mask = lung_mask.astype(np.uint8)

        print(f"肺部阈值内体素: {np.sum(lung_mask)}/{lung_mask.size}")

        # 标记连通区域
        labeled_mask, num_features = ndimage.label(lung_mask)

        if num_features == 0:
            raise ValueError("未找到肺部连通区域，请调整肺部阈值参数")

        print(f"找到 {num_features} 个肺部连通区域")

        # 计算每个区域大小并排序
        region_sizes = []
        for i in range(1, num_features + 1):
            size = np.sum(labeled_mask == i)
            region_sizes.append((i, size))

        region_sizes.sort(key=lambda x: x[1], reverse=True)

        # 选择最大的两个区域（左右肺）
        if len(region_sizes) < 2:
            print("警告: 只找到一个肺部区域，可能不完整")
            lung_regions = [region_sizes[0][0]] if region_sizes else []
        else:
            # 取前两个最大的区域作为左右肺
            lung_regions = [region_sizes[0][0], region_sizes[1][0]]
            print(f"选择区域 {lung_regions} 作为左右肺")

        # 创建肺部掩码
        final_lung_mask = np.zeros_like(data, dtype=np.uint8)
        for region_label in lung_regions:
            final_lung_mask = np.logical_or(final_lung_mask,
                                            labeled_mask == region_label)

        lung_voxels = np.sum(final_lung_mask)
        print(f"提取肺部区域体素数: {lung_voxels}")

        return final_lung_mask.astype(np.uint8)

    def extract_heart(self, data):
        """
        提取心脏区域
        """
        print("步骤2: 提取心脏区域...")

        # 设置心脏阈值上限
        if self.heart_window_max is None:
            heart_window_max = np.max(data)
        else:
            heart_window_max = self.heart_window_max

        # 创建心脏二值掩码
        heart_mask = np.logical_and(data >= self.heart_window_min,
                                    data <= heart_window_max)
        heart_mask = heart_mask.astype(np.uint8)

        print(f"心脏阈值内体素: {np.sum(heart_mask)}")

        # 标记连通区域
        labeled_mask, num_features = ndimage.label(heart_mask)

        if num_features == 0:
            raise ValueError("未找到心脏连通区域，请调整心脏阈值参数")

        print(f"找到 {num_features} 个心脏连通区域")

        # 计算每个区域大小并排序
        region_sizes = []
        for i in range(1, num_features + 1):
            size = np.sum(labeled_mask == i)
            region_sizes.append((i, size))

        region_sizes.sort(key=lambda x: x[1], reverse=True)

        # 选择最大的连通区域（心脏）
        heart_region = region_sizes[0][0]
        print(f"选择区域 {heart_region} 作为心脏区域")

        # 创建心脏掩码
        heart_mask = (labeled_mask == heart_region).astype(np.uint8)
        heart_voxels = np.sum(heart_mask)
        print(f"心脏区域体素数: {heart_voxels}")

        return heart_mask

    def combine_regions(self, data, lung_mask, heart_mask):
        """
        合并肺部和心脏区域
        """
        print("步骤3: 合并肺部和心脏区域...")

        # 合并掩码（肺部 + 心脏）
        combined_mask = np.logical_or(lung_mask, heart_mask)

        # 应用掩码，保留目标区域，其他区域设为背景值
        background_value = -1000  # CT背景典型值
        combined_data = data.copy()
        combined_data[~combined_mask] = background_value

        # 统计信息
        total_voxels = data.size
        preserved_voxels = np.sum(combined_mask)
        preservation_ratio = preserved_voxels / total_voxels

        print(f"合并后统计:")
        print(f"  - 保留体素数: {preserved_voxels}/{total_voxels} ({preservation_ratio:.2%})")
        print(f"  - 肺部区域: {np.sum(lung_mask)} 体素")
        print(f"  - 心脏区域: {np.sum(heart_mask)} 体素")
        print(f"  - 最终HU范围: [{np.min(combined_data):.1f}, {np.max(combined_data):.1f}]")

        return combined_data, combined_mask

    def process_single_image(self, input_path, output_path):
        """
        处理单个图像文件
        """
        start_time = datetime.now()

        try:
            print(f"\n处理图像: {input_path}")

            # 加载图像
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"输入文件不存在: {input_path}")

            img = nib.load(input_path)
            data = img.get_fdata()
            affine = img.affine
            header = img.header

            print(f"图像信息: 尺寸{data.shape}, HU范围[{np.min(data):.1f}, {np.max(data):.1f}]")

            # 步骤1: 提取肺部
            lung_mask = self.extract_lungs(data)

            # 步骤2: 提取心脏
            heart_mask = self.extract_heart(data)

            # 步骤3: 合并区域
            combined_data, combined_mask = self.combine_regions(data, lung_mask, heart_mask)

            # 保存结果
            combined_img = nib.Nifti1Image(combined_data.astype(np.float32),
                                           affine, header)

            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            nib.save(combined_img, output_path)

            processing_time = (datetime.now() - start_time).total_seconds()

            print(f"处理完成: {output_path} (耗时: {processing_time:.2f}秒)")

            return {
                'status': 'success',
                'input_path': input_path,
                'output_path': output_path,
                'processing_time': processing_time,
                'lung_voxels': int(np.sum(lung_mask)),
                'heart_voxels': int(np.sum(heart_mask)),
                'total_preserved': int(np.sum(combined_mask)),
                'final_hu_range': [float(np.min(combined_data)),
                                   float(np.max(combined_data))]
            }

        except Exception as e:
            error_result = {
                'status': 'error',
                'input_path': input_path,
                'error_message': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            print(f"处理失败 {input_path}: {str(e)}")
            return error_result

    def visualize_results(self, original_data, combined_data, slice_index=None):
        """
        可视化结果
        """
        if slice_index is None:
            slice_index = original_data.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(original_data[:, :, slice_index], cmap='gray',
                       vmin=-1000, vmax=400)
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        # 处理后的图像
        axes[1].imshow(combined_data[:, :, slice_index], cmap='gray',
                       vmin=-1000, vmax=400)
        axes[1].set_title('心肺联合提取')
        axes[1].axis('off')

        # 差异图像
        diff = combined_data[:, :, slice_index] - original_data[:, :, slice_index]
        axes[2].imshow(diff, cmap='coolwarm', vmin=-500, vmax=500)
        axes[2].set_title('差异图像')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


def batch_process(input_dir, output_dir, **kwargs):
    """
    批量处理目录中的所有NIfTI文件
    """
    # 创建提取器实例
    extractor = CardioPulmonaryExtractor(**kwargs)

    # 查找NIfTI文件
    nifti_files = find_nifti_files(input_dir)

    if not nifti_files:
        print(f"错误: 在目录 {input_dir} 中未找到NIfTI文件")
        return

    print(f"找到 {len(nifti_files)} 个NIfTI文件")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    # 处理每个文件
    for input_file in tqdm(nifti_files, desc="处理进度"):
        try:
            # 生成输出文件名
            stem = Path(input_file).stem
            if Path(input_file).suffix == '.gz':
                stem = Path(input_file).stem.split('.')[0]

            output_filename = f"cardiopulmonary_{stem}.nii.gz"
            output_path = Path(output_dir) / output_filename

            # 处理图像
            result = extractor.process_single_image(str(input_file), str(output_path))

            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1

        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            failed += 1

    # 输出统计报告
    print("\n" + "=" * 60)
    print("批量处理统计报告")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理文件总数: {len(nifti_files)}")
    print(f"成功处理: {successful} 个文件")
    print(f"处理失败: {failed} 个文件")
    print("=" * 60)


def find_nifti_files(input_dir):
    """
    查找目录中的所有NIfTI文件
    """
    input_path = Path(input_dir)
    nifti_extensions = ['*.nii', '*.nii.gz', '*.hdr', '*.img']
    nifti_files = []

    for ext in nifti_extensions:
        nifti_files.extend(input_path.rglob(ext))

    return sorted(list(set(nifti_files)))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='心肺联合区域提取程序 - 提取肺部+心脏区域（包含中心大血管）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python cardiopulmonary_extractor.py -i ./input -o ./output

  # 自定义参数
  python cardiopulmonary_extractor.py -i ./ct_scans -o ./results \\
    --lung_min -1000 --lung_max -400 \\
    --heart_min -180

说明:
  - 程序先提取肺部区域（左右肺）
  - 再提取心脏区域（包含中心大血管）
  - 最后合并两个区域，保留肺部和心脏之间的血管连接
  - 这样可以在后续分割中处理中心大血管区域
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='输入目录路径')
    parser.add_argument('--output', '-o', required=True,
                        help='输出目录路径')
    parser.add_argument('--lung_min', type=float, default=-1000,
                        help='肺部提取下限(HU)，默认-1000')
    parser.add_argument('--lung_max', type=float, default=-400,
                        help='肺部提取上限(HU)，默认-400')
    parser.add_argument('--heart_min', type=float, default=-180,
                        help='心脏提取下限(HU)，默认-180')
    parser.add_argument('--heart_max', type=float, default=None,
                        help='心脏提取上限(HU)，默认None（使用图像最大值）')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return

    print("=" * 70)
    print("心肺联合区域提取程序")
    print("=" * 70)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"肺部窗口: [{args.lung_min}, {args.lung_max}] HU")
    print(f"心脏窗口: [{args.heart_min}, {args.heart_max or 'auto'}] HU")
    print("=" * 70)

    # 参数配置
    params = {
        'lung_window_min': args.lung_min,
        'lung_window_max': args.lung_max,
        'heart_window_min': args.heart_min,
        'heart_window_max': args.heart_max
    }

    try:
        # 执行批量处理
        batch_process(args.input, args.output, **params)
        print("\n处理完成!")

    except Exception as e:
        print(f"程序执行错误: {str(e)}")


if __name__ == "__main__":
    main()