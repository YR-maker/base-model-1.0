import numpy as np
import nibabel as nib
from scipy import ndimage
import os
import argparse


def extract_heart_roi(input_path, output_path, threshold_min=100, threshold_max=None):
    """
    提取心脏ROI区域

    参数:
    - input_path: 输入nii.gz文件路径
    - output_path: 输出nii.gz文件路径
    - threshold_min: 阈值下限，默认-100 HU
    - threshold_max: 阈值上限，默认None（使用图像最大值）
    """

    # 读取nii.gz文件
    print(f"读取文件: {input_path}")
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # 获取图像信息
    original_shape = data.shape
    print(f"原始图像尺寸: {original_shape}")
    print(f"HU值范围: [{np.min(data):.2f}, {np.max(data):.2f}]")

    # 设置阈值上限（如果未指定则使用图像最大值）
    if threshold_max is None:
        threshold_max = np.max(data)

    # 创建二值掩码：阈值范围内的体素设为1，其他为0
    binary_mask = np.logical_and(data >= threshold_min, data <= threshold_max)
    binary_mask = binary_mask.astype(np.uint8)

    print(f"阈值范围: [{threshold_min}, {threshold_max}]")
    print(f"阈值内体素数量: {np.sum(binary_mask)}/{binary_mask.size}")

    # 标记连通区域
    labeled_mask, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        raise ValueError("未找到任何连通区域，请调整阈值参数")

    print(f"找到 {num_features} 个连通区域")

    # 计算每个连通区域的大小
    region_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_mask == i)
        region_sizes.append((i, size))

    # 按大小排序，选择最大的连通区域（心脏）
    region_sizes.sort(key=lambda x: x[1], reverse=True)
    largest_region_label = region_sizes[0][0]

    print(f"最大连通区域标签: {largest_region_label}, 体素数: {region_sizes[0][1]}")

    # 创建心脏ROI掩码
    heart_mask = (labeled_mask == largest_region_label).astype(np.uint8)

    # 应用掩码：只保留心脏区域，其他区域设为背景值（通常为-1000或图像最小值）
    background_value = np.min(data)  # 或者设置为固定的背景值，如-1000
    heart_roi_data = data * heart_mask + background_value * (1 - heart_mask)

    print(f"心脏ROI尺寸: {heart_roi_data.shape}")
    print(f"心脏ROI HU值范围: [{np.min(heart_roi_data):.2f}, {np.max(heart_roi_data):.2f}]")

    # 创建新的nii图像
    heart_roi_img = nib.Nifti1Image(heart_roi_data, affine, header)

    # 保存结果
    nib.save(heart_roi_img, output_path)
    print(f"心脏ROI已保存至: {output_path}")

    return heart_roi_data, heart_mask


def process_directory(input_dir, output_dir, threshold_min=-100, threshold_max=None):
    """
    处理整个目录下的nii.gz文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz') or filename.endswith('.nii'):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"heart_roi_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            try:
                heart_roi, mask = extract_heart_roi(input_path, output_path,
                                                    threshold_min, threshold_max)
                processed_files.append(filename)
                print(f"成功处理: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    print(f"\n处理完成! 共处理 {len(processed_files)} 个文件")
    return processed_files


def visualize_slice(image_data, mask_data, slice_index=None, axis=2):
    """
    可视化一个切片（可选功能）
    """
    import matplotlib.pyplot as plt

    if slice_index is None:
        slice_index = image_data.shape[axis] // 2

    if axis == 0:
        original_slice = image_data[slice_index, :, :]
        mask_slice = mask_data[slice_index, :, :]
    elif axis == 1:
        original_slice = image_data[:, slice_index, :]
        mask_slice = mask_data[:, slice_index, :]
    else:
        original_slice = image_data[:, :, slice_index]
        mask_slice = mask_data[:, :, slice_index]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    axes[1].imshow(mask_slice, cmap='gray')
    axes[1].set_title('心脏掩码')
    axes[1].axis('off')

    # 叠加显示
    axes[2].imshow(original_slice, cmap='gray')
    axes[2].imshow(mask_slice, cmap='jet', alpha=0.3)
    axes[2].set_title('叠加显示')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='提取心脏ROI区域')
    parser.add_argument('--input', '-i', type=str,
                        default="/home/yangrui/Project/Base-models/datasets/AAA-datapre/ROI/imageCAS/data/img/2_img.nii.gz",
                        help='输入nii.gz文件路径或目录路径')
    parser.add_argument('--output', '-o', type=str,
                        default="/home/yangrui/Project/Base-models/datasets/AAA-datapre/ROI/imageCAS/data/output/2_HU.nii.gz",
                        help='输出nii.gz文件路径或目录路径')
    parser.add_argument('--threshold_min', type=float, default=-200,
                        help='阈值下限，默认-100 HU')
    parser.add_argument('--threshold_max', type=float, default=None,
                        help='阈值上限，默认None（使用图像最大值）')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')

    args = parser.parse_args()

    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        heart_roi, mask = extract_heart_roi(
            args.input,
            args.output,
            args.threshold_min,
            args.threshold_max
        )

        if args.visualize:
            visualize_slice(nib.load(args.input).get_fdata(), mask)

    elif os.path.isdir(args.input):
        # 处理目录
        processed_files = process_directory(
            args.input,
            args.output,
            args.threshold_min,
            args.threshold_max
        )
    else:
        print("错误: 输入路径不存在")