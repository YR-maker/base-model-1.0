import nibabel as nib


def get_z_axis_thickness(nii_file_path):
    """
    获取nii.gz文件的Z轴切片厚度

    参数:
    nii_file_path: nii.gz文件的路径

    返回:
    z_thickness: Z轴切片厚度（单位：毫米）
    """
    # 加载NIfTI文件
    img = nib.load(nii_file_path)

    # 获取头信息中的像素尺寸
    # pixdim[1], pixdim[2], pixdim[3] 分别对应x, y, z轴的分辨率
    pixdim = img.header['pixdim']

    # Z轴分辨率（切片厚度）是pixdim[3]
    z_thickness = pixdim[3]

    return z_thickness


def get_detailed_image_info(nii_file_path):
    """
    获取nii.gz文件的详细信息
    """
    # 加载NIfTI文件
    img = nib.load(nii_file_path)

    # 获取图像数据
    data = img.get_fdata()

    # 获取图像尺寸
    height, width, depth = data.shape

    # 获取像素尺寸信息
    pixdim = img.header['pixdim']

    print("=== NIfTI图像详细信息 ===")
    print(f"文件路径: {nii_file_path}")
    print(f"图像尺寸 (高度×宽度×层数): {height} × {width} × {depth}")
    print(f"X轴分辨率: {pixdim[1]:.6f} mm")
    print(f"Y轴分辨率: {pixdim[2]:.6f} mm")
    print(f"Z轴切片厚度: {pixdim[3]:.6f} mm")
    print(f"图像值范围: [{data.min():.2f}, {data.max():.2f}]")

    # 计算实际物理尺寸
    x_range = pixdim[1] * height
    y_range = pixdim[2] * width
    z_range = pixdim[3] * depth

    print(f"实际扫描范围: {x_range:.2f} × {y_range:.2f} × {z_range:.2f} mm")

    return {
        'x_resolution': pixdim[1],
        'y_resolution': pixdim[2],
        'z_thickness': pixdim[3],
        'dimensions': (height, width, depth),
        'value_range': (data.min(), data.max())
    }


# 使用示例
if __name__ == "__main__":
    # 替换为你的nii.gz文件路径
    file_path = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-clip/all/372/372.img.nii.gz"  # 请修改为实际文件路径

    try:
        # 方法1: 仅获取Z轴厚度
        thickness = get_z_axis_thickness(file_path)
        print(f"Z轴切片厚度: {thickness:.6f} mm")

        print("\n" + "=" * 50 + "\n")

        # 方法2: 获取完整图像信息
        info = get_detailed_image_info(file_path)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确保文件路径正确，且文件存在")
    except Exception as e:
        print(f"处理文件时出错: {e}")