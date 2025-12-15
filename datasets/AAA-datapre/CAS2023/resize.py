import os
import glob
import SimpleITK as sitk

# ================= 核心配置 =================
INPUT_DIR = "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-reshape/val"
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-reshape/val-resize"

# Z轴放大倍数
Z_FACTOR = 2.0


# ================= 辅助函数 =================

def save_nifti_safe(image_obj, final_path):
    """
    安全保存函数：先存为临时文件，再重命名。
    解决 SimpleITK 对特殊文件名的报错问题。
    """
    dirname = os.path.dirname(final_path)
    filename = os.path.basename(final_path)

    # 临时文件名 TEMP_xxxx.nii.gz
    temp_filename = "TEMP_" + filename.replace(".", "_") + ".nii.gz"
    temp_path = os.path.join(dirname, temp_filename)

    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(temp_path)
        writer.SetImageIO("NiftiImageIO")
        writer.Execute(image_obj)

        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)
    except Exception as e:
        print(f"      [Error] Failed to save {filename}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def upsample_z_axis(itk_img, factor):
    """
    仅对 Z 轴进行重采样
    """
    orig_size = itk_img.GetSize()
    orig_spacing = itk_img.GetSpacing()
    orig_origin = itk_img.GetOrigin()
    orig_direction = itk_img.GetDirection()

    # 新尺寸：Z * factor
    new_size = [orig_size[0], orig_size[1], int(orig_size[2] * factor)]
    # 新间距：Z / factor
    new_spacing = [orig_spacing[0], orig_spacing[1], orig_spacing[2] / factor]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(orig_origin)
    resampler.SetOutputDirection(orig_direction)

    return resampler


# ================= 主逻辑 =================

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[Error] Input directory not found: {INPUT_DIR}")
        return

    print(f"Source Root: {INPUT_DIR}")
    print(f"Target Root: {OUTPUT_DIR}")
    print(f"Z-Axis Scale Factor: {Z_FACTOR}")

    # 1. 获取所有子文件夹 (例如: 001, 002, 003...)
    # 过滤掉非文件夹的杂项文件
    case_folders = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ])

    print(f"Found {len(case_folders)} case folders.")

    for case_id in case_folders:
        case_in_dir = os.path.join(INPUT_DIR, case_id)
        case_out_dir = os.path.join(OUTPUT_DIR, case_id)

        # 2. 在输出目录创建对应的子文件夹
        os.makedirs(case_out_dir, exist_ok=True)

        # 3. 查找该子文件夹下的所有 .nii.gz 文件
        nii_files = sorted(glob.glob(os.path.join(case_in_dir, "*.nii.gz")))

        if not nii_files:
            print(f"   [Skip] No .nii.gz files in folder: {case_id}")
            continue

        print(f"\n>>> Processing Folder: {case_id} ({len(nii_files)} files)")

        for file_path in nii_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(case_out_dir, filename)

            # 判断是否为标签 (Label)
            fname_lower = filename.lower()
            is_label = ("label" in fname_lower) or ("seg" in fname_lower) or ("mask" in fname_lower)

            try:
                # 读取
                img_obj = sitk.ReadImage(file_path)

                # 设置重采样器
                resampler = upsample_z_axis(img_obj, Z_FACTOR)

                # 插值策略选择
                if is_label:
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 标签用最近邻
                else:
                    resampler.SetInterpolator(sitk.sitkLinear)  # 图像用线性

                # 执行
                new_img_obj = resampler.Execute(img_obj)

                # 保存
                save_nifti_safe(new_img_obj, target_path)
                print(f"      Processed: {filename} ({'Label' if is_label else 'Image'})")

            except Exception as e:
                print(f"      [Error] Failed: {filename} -> {e}")

    print("\n[Done] All tasks finished.")


if __name__ == "__main__":
    main()