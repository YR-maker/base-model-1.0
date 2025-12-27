import os
import glob
import SimpleITK as sitk
import numpy as np

# ================= 核心配置 =================
INPUT_DIR = "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-reshape/all"
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-resize/all"

# 目标尺寸 (格式: Depth/Z, Height/Y, Width/X)
TARGET_SHAPE_ZYX = (300, 640, 640)


# ================= 辅助函数 =================

def save_nifti_safe(image_obj, final_path):
    dirname = os.path.dirname(final_path)
    filename = os.path.basename(final_path)
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


def resize_to_fixed_size(itk_img, target_zyx, is_label=False):
    """
    将图像重采样到固定的 (Z, Y, X) 尺寸。
    参数:
        is_label: 如果是 True，强制使用最近邻插值；如果是 False，使用 B-Spline 插值。
    """
    # SimpleITK 内部尺寸顺序是 (X, Y, Z)
    target_size_sitk = [int(target_zyx[2]), int(target_zyx[1]), int(target_zyx[0])]

    orig_size = itk_img.GetSize()  # (X, Y, Z)
    orig_spacing = itk_img.GetSpacing()  # (X, Y, Z)
    orig_origin = itk_img.GetOrigin()
    orig_direction = itk_img.GetDirection()

    # 计算新的 Spacing 以保持物理体积一致
    new_spacing = [
        orig_spacing[0] * (orig_size[0] / target_size_sitk[0]),
        orig_spacing[1] * (orig_size[1] / target_size_sitk[1]),
        orig_spacing[2] * (orig_size[2] / target_size_sitk[2])
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(target_size_sitk)
    resampler.SetOutputOrigin(orig_origin)
    resampler.SetOutputDirection(orig_direction)

    # ================= 核心修改点 =================
    if is_label:
        # 标签：必须是最近邻，保证整数类别不变 (0, 1, 2...)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # 图像：使用 B-Spline (3阶)，比 Linear 更平滑，质量更高
        # 注意：B-Spline 计算量比 Linear 大，速度会稍慢一点，但质量最好
        resampler.SetInterpolator(sitk.sitkBSpline)
        # 如果觉得 B-Spline 太慢，也可以试 sitk.sitkBlackmanWindowedSinc (更锐利)
    # ============================================

    return resampler


# ================= 主逻辑 =================

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[Error] Input directory not found: {INPUT_DIR}")
        return

    print(f"Source Root: {INPUT_DIR}")
    print(f"Target Root: {OUTPUT_DIR}")
    print(f"Target Shape (Z, Y, X): {TARGET_SHAPE_ZYX}")
    print("Interpolation: Image -> B-Spline (Smooth), Label -> NearestNeighbor")

    case_folders = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ])

    print(f"Found {len(case_folders)} case folders.")

    for case_id in case_folders:
        case_in_dir = os.path.join(INPUT_DIR, case_id)
        case_out_dir = os.path.join(OUTPUT_DIR, case_id)

        os.makedirs(case_out_dir, exist_ok=True)
        nii_files = sorted(glob.glob(os.path.join(case_in_dir, "*.nii.gz")))

        if not nii_files:
            continue

        print(f"\n>>> Processing Folder: {case_id} ({len(nii_files)} files)")

        for file_path in nii_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(case_out_dir, filename)

            # 自动识别是否为标签
            fname_lower = filename.lower()
            is_label = ("label" in fname_lower) or ("seg" in fname_lower) or ("mask" in fname_lower)

            try:
                img_obj = sitk.ReadImage(file_path)

                # 传入 is_label 参数，决定插值方式
                resampler = resize_to_fixed_size(img_obj, TARGET_SHAPE_ZYX, is_label=is_label)

                new_img_obj = resampler.Execute(img_obj)

                save_nifti_safe(new_img_obj, target_path)

                method_str = "NearestNeighbor" if is_label else "B-Spline"
                print(f"      [{method_str}] Processed: {filename}")

            except Exception as e:
                print(f"      [Error] Failed: {filename} -> {e}")

    print("\n[Done] All tasks finished.")


if __name__ == "__main__":
    main()