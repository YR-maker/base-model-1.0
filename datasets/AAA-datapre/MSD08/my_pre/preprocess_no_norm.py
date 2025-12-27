import os
import shutil
import glob
import re
import numpy as np
import SimpleITK as sitk

# ================= 配置区域 =================

# 1. 输入路径 (原始数据)
SRC_IMG_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/imagesTr"
SRC_LABEL_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/reannotated_fixed"

# 2. 输出路径
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-61-1/all"

# 3. 筛选标准 (只处理层厚 < 2.0mm 的数据)
THIN_SLICE_THRESHOLD = 2.0

# 4. 标签前缀 (用于匹配文件名)
SRC_LABEL_PREFIX = "hp"

# 5. 固定截断设置 (Fixed Clipping)
ENABLE_CLIPPING = True  # 开关
CLIP_MIN = -200.0  # 下界
CLIP_MAX = 400.0  # 上界

# 6. 【新增】Z轴两倍缩放设置 (Z-axis 2x Scaling / Upsampling)
#    开启后，Z轴层数变为原来的2倍，层厚(spacing)变为原来的1/2
ENABLE_Z_RESCALE = False


# ================= 核心函数 =================

def extract_id(filename):
    """从文件名中提取数字 ID"""
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def save_nifti_safe(image_obj, final_path):
    """安全保存函数：防止生成 .hdr/.img"""
    final_path = str(final_path)
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
        shutil.move(temp_path, final_path)

        # 清理垃圾文件
        junk_base = final_path.replace(".nii.gz", "")
        for ext in [".hdr", ".img"]:
            junk_file = junk_base + ext
            if os.path.exists(junk_file):
                os.remove(junk_file)

    except Exception as e:
        print(f"      ❌ 保存失败: {filename} -> {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def resample_z_axis_x2(itk_image, is_label=False):
    """
    对 SimpleITK 对象进行 Z 轴 2 倍上采样 (层数x2, Spacing/2)
    :param itk_image: 原始 ITK 图像对象
    :param is_label: 是否为标签 (标签必须用最近邻插值)
    """
    # 1. 获取原始信息
    orig_spacing = itk_image.GetSpacing()
    orig_size = itk_image.GetSize()

    # 2. 计算新的 Spacing (Z轴减半)
    # spacing: (x, y, z)
    new_spacing = (orig_spacing[0], orig_spacing[1], orig_spacing[2] * 0.5)

    # 3. 计算新的 Size (Z轴加倍)
    # 保持物理尺寸不变: new_size = orig_size * (orig_spacing / new_spacing)
    new_size = [
        int(orig_size[0]),
        int(orig_size[1]),
        int(round(orig_size[2] * (orig_spacing[2] / new_spacing[2])))  # 约等于 orig_size[2] * 2
    ]

    # 4. 构建重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    # 关键：方向和原点必须保持一致
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    # 5. 设置插值方式
    if is_label:
        # 标签绝对不能用线性插值，否则会出现 0.5 这种小数
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # 图像通常使用线性插值 (sitkLinear) 或 B样条 (sitkBSpline)
        # 这里使用 Linear 速度快且对 CT 足够
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(itk_image)


def process_single_case(img_path, lbl_path, output_folder, case_id):
    """处理单个样本：读取 -> 检查 -> 裁剪 -> [重采样] -> 截断 -> 保存"""

    # 1. 读取图像和标签
    image = sitk.ReadImage(img_path)
    label = sitk.ReadImage(lbl_path)

    # 2. 【核心筛选】检查层厚
    spacing = image.GetSpacing()
    z_spacing = spacing[2]

    if z_spacing > THIN_SLICE_THRESHOLD:
        return False, f"Skip (Thick slice: {z_spacing:.2f}mm)"

    # 3. 【裁剪】基于前景的 ROI Crop
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    binary_label = sitk.BinaryThreshold(label, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    label_stats.Execute(binary_label)

    if not label_stats.HasLabel(1):
        return False, "Skip (Empty Label)"

    bbox = label_stats.GetBoundingBox(1)

    # 执行裁剪
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetRegionOfInterest(bbox)

    cropped_image_obj = roi_filter.Execute(image)
    cropped_label_obj = roi_filter.Execute(label)

    # =======================================================
    # 4. 【新增】Z 轴两倍缩放 (Resampling)
    # =======================================================
    if ENABLE_Z_RESCALE:
        # print(f"      ... Resampling Z-axis (Original Z: {cropped_image_obj.GetSize()[2]})")
        processed_image_obj = resample_z_axis_x2(cropped_image_obj, is_label=False)
        processed_label_obj = resample_z_axis_x2(cropped_label_obj, is_label=True)
        rescale_msg = f"Z-Rescaled (Z: {cropped_image_obj.GetSize()[2]}->{processed_image_obj.GetSize()[2]})"
    else:
        processed_image_obj = cropped_image_obj
        processed_label_obj = cropped_label_obj
        rescale_msg = "No Rescale"
    # =======================================================

    # 5. 转为 Numpy 进行像素处理
    img_arr = sitk.GetArrayFromImage(processed_image_obj)
    lbl_arr = sitk.GetArrayFromImage(processed_label_obj)

    # 6. 【标签处理】二值化
    new_lbl_arr = np.zeros_like(lbl_arr)
    new_lbl_arr[lbl_arr > 0] = 1

    # 7. 【图像处理】固定截断 (Fixed Clipping)
    img_arr = img_arr.astype(np.float32)

    if ENABLE_CLIPPING:
        img_arr = np.clip(img_arr, CLIP_MIN, CLIP_MAX)
        clip_msg = f"Clipped [{CLIP_MIN}, {CLIP_MAX}]"
    else:
        clip_msg = "No Clip"

    new_img_arr = img_arr

    # 8. 转回 SimpleITK 对象
    final_img_obj = sitk.GetImageFromArray(new_img_arr)
    final_img_obj.CopyInformation(processed_image_obj)  # 复制重采样后的信息

    final_lbl_obj = sitk.GetImageFromArray(new_lbl_arr.astype(np.uint8))
    final_lbl_obj.CopyInformation(processed_label_obj)  # 复制重采样后的信息

    # 9. 保存
    case_dir = os.path.join(output_folder, case_id)
    os.makedirs(case_dir, exist_ok=True)

    target_img_name = f"{case_id}.img.nii.gz"
    target_lbl_name = f"{case_id}.label.nii.gz"

    save_nifti_safe(final_img_obj, os.path.join(case_dir, target_img_name))
    save_nifti_safe(final_lbl_obj, os.path.join(case_dir, target_lbl_name))

    return True, f"Success ({clip_msg}, {rescale_msg}, Shape: {new_img_arr.shape})"


# ================= 主程序 =================

def main():
    if not os.path.exists(SRC_IMG_DIR):
        print(f"❌ 源目录不存在: {SRC_IMG_DIR}")
        return

    img_files = sorted(glob.glob(os.path.join(SRC_IMG_DIR, "hepaticvessel_*.nii.gz")))

    print(f"🔍 扫描目录: {SRC_IMG_DIR}")
    print(f"📄 找到文件: {len(img_files)} 个")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"⚙️ 筛选条件: 层厚 < {THIN_SLICE_THRESHOLD} mm")
    if ENABLE_CLIPPING:
        print(f"✂️ 固定截断: 开启 (范围: {CLIP_MIN} ~ {CLIP_MAX})")

    if ENABLE_Z_RESCALE:
        print(f"📏 Z轴缩放: 开启 (2x Upsampling, Spacing/2)")
    else:
        print(f"📏 Z轴缩放: 关闭")

    print("-" * 60)

    count_processed = 0
    count_skipped_thick = 0

    for img_path in img_files:
        filename = os.path.basename(img_path)
        case_id = extract_id(filename)

        if not case_id:
            continue

        lbl_name = f"{SRC_LABEL_PREFIX}{case_id}.nii.gz"
        lbl_path = os.path.join(SRC_LABEL_DIR, lbl_name)

        if not os.path.exists(lbl_path):
            print(f"⚠️  [ID: {case_id}] 缺失标签文件，跳过")
            continue

        print(f"⏳ [ID: {case_id}] 处理中...", end="\r")
        success, msg = process_single_case(img_path, lbl_path, OUTPUT_DIR, case_id)

        if success:
            print(f"✅ [ID: {case_id}] {msg}")
            count_processed += 1
        else:
            if "Thick slice" in msg:
                count_skipped_thick += 1
            else:
                print(f"❌ [ID: {case_id}] {msg}")

    print("-" * 60)
    print(f"🎉 全部完成！")
    print(f"📥 总输入文件: {len(img_files)}")
    print(f"⏭️ 跳过厚层数据: {count_skipped_thick}")
    print(f"💾 成功处理并保存: {count_processed}")
    print(f"📂 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()