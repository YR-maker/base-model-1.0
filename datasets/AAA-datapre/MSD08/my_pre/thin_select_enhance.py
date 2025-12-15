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
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-61/all"

# 3. 筛选标准 (只处理层厚 < 2.0mm 的数据)
THIN_SLICE_THRESHOLD = 2.0

# 4. 标签前缀 (用于匹配文件名)
# 假设图片是 hepaticvessel_001.nii.gz，标签是 hp001.nii.gz
SRC_LABEL_PREFIX = "hp"


# ================= 核心函数 =================

def extract_id(filename):
    """从文件名中提取数字 ID"""
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def save_nifti_safe(image_obj, final_path):
    """
    安全保存函数：
    先保存为临时文件 (temp.nii.gz)，然后重命名。
    这能防止 SimpleITK 因为文件名含 .img. 而误生成 Analyze 格式 (.hdr/.img)
    """
    final_path = str(final_path)  # 确保是字符串
    dirname = os.path.dirname(final_path)
    filename = os.path.basename(final_path)

    # 临时文件名
    temp_filename = "TEMP_" + filename.replace(".", "_") + ".nii.gz"
    temp_path = os.path.join(dirname, temp_filename)

    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(temp_path)
        writer.SetImageIO("NiftiImageIO")  # 强制 NIfTI
        writer.Execute(image_obj)

        if os.path.exists(final_path):
            os.remove(final_path)

        # 重命名
        shutil.move(temp_path, final_path)

        # 双重检查：清理可能产生的垃圾 hdr/img 文件
        junk_base = final_path.replace(".nii.gz", "")
        for ext in [".hdr", ".img"]:
            junk_file = junk_base + ext
            if os.path.exists(junk_file):
                os.remove(junk_file)

    except Exception as e:
        print(f"      ❌ 保存失败: {filename} -> {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_intensity(img_arr):
    """
    执行 20-98 分位数截断，并归一化到 [0, 1]
    """
    lower = np.percentile(img_arr, 20)
    upper = np.percentile(img_arr, 98)

    # 截断 (Clip)
    img_arr = np.clip(img_arr, lower, upper)

    # 归一化 (0-1)
    if upper != lower:
        img_arr = (img_arr - lower) / (upper - lower)
    else:
        img_arr[:] = 0  # 避免除以0

    return img_arr


def process_single_case(img_path, lbl_path, output_folder, case_id):
    """处理单个样本：读取 -> 检查厚度 -> 裁剪 -> 归一化 -> 保存"""

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
    # 确保标签是二值的用于计算bbox (虽然本来可能就是，但安全起见)
    binary_label = sitk.BinaryThreshold(label, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    label_stats.Execute(binary_label)

    if not label_stats.HasLabel(1):
        return False, "Skip (Empty Label)"

    bbox = label_stats.GetBoundingBox(1)  # (x, y, z, w, h, d)

    # 执行裁剪
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetRegionOfInterest(bbox)

    cropped_image_obj = roi_filter.Execute(image)
    cropped_label_obj = roi_filter.Execute(label)

    # 4. 转为 Numpy 进行像素处理
    img_arr = sitk.GetArrayFromImage(cropped_image_obj)
    lbl_arr = sitk.GetArrayFromImage(cropped_label_obj)

    # 5. 【标签处理】二值化 (所有非0都变1)
    # 注意：原始标签可能有动脉/静脉区分，这里统一合并为血管(1)
    new_lbl_arr = np.zeros_like(lbl_arr)
    new_lbl_arr[lbl_arr > 0] = 1

    # 6. 【图像处理】归一化 (20-98% -> 0-1)
    new_img_arr = normalize_intensity(img_arr)

    # 7. 转回 SimpleITK 对象
    # 图像
    final_img_obj = sitk.GetImageFromArray(new_img_arr)
    final_img_obj.CopyInformation(cropped_image_obj)  # 关键：继承裁剪后的空间信息

    # 标签
    final_lbl_obj = sitk.GetImageFromArray(new_lbl_arr.astype(np.uint8))
    final_lbl_obj.CopyInformation(cropped_label_obj)  # 关键：继承裁剪后的空间信息

    # 8. 保存
    # 创建对应的 ID 文件夹 (例如: .../all/001/)
    case_dir = os.path.join(output_folder, case_id)
    os.makedirs(case_dir, exist_ok=True)

    # 目标文件名 (保持 ID 不变)
    target_img_name = f"{case_id}.img.nii.gz"
    target_lbl_name = f"{case_id}.label.nii.gz"

    save_nifti_safe(final_img_obj, os.path.join(case_dir, target_img_name))
    save_nifti_safe(final_lbl_obj, os.path.join(case_dir, target_lbl_name))

    return True, f"Success ({z_spacing:.2f}mm, Crop shape: {new_img_arr.shape})"


# ================= 主程序 =================

def main():
    if not os.path.exists(SRC_IMG_DIR):
        print(f"❌ 源目录不存在: {SRC_IMG_DIR}")
        return

    # 获取所有源图像
    img_files = sorted(glob.glob(os.path.join(SRC_IMG_DIR, "hepaticvessel_*.nii.gz")))

    print(f"🔍 扫描目录: {SRC_IMG_DIR}")
    print(f"📄 找到文件: {len(img_files)} 个")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"⚙️ 筛选条件: 层厚 < {THIN_SLICE_THRESHOLD} mm")
    print("-" * 60)

    count_processed = 0
    count_skipped_thick = 0

    for img_path in img_files:
        filename = os.path.basename(img_path)
        case_id = extract_id(filename)  # 提取 ID，例如 "001"

        if not case_id:
            continue

        # 寻找对应的 label 文件
        # 假设文件名格式为 hp001.nii.gz
        lbl_name = f"{SRC_LABEL_PREFIX}{case_id}.nii.gz"
        lbl_path = os.path.join(SRC_LABEL_DIR, lbl_name)

        if not os.path.exists(lbl_path):
            print(f"⚠️  [ID: {case_id}] 缺失标签文件，跳过")
            continue

        # 开始处理
        print(f"⏳ [ID: {case_id}] 处理中...", end="\r")
        success, msg = process_single_case(img_path, lbl_path, OUTPUT_DIR, case_id)

        if success:
            print(f"✅ [ID: {case_id}] {msg}")
            count_processed += 1
        else:
            if "Thick slice" in msg:
                count_skipped_thick += 1
                # 也可以选择不打印由厚层导致的跳过，保持清爽
                # print(f"⚪ [ID: {case_id}] {msg}")
            else:
                print(f"❌ [ID: {case_id}] {msg}")

    print("-" * 60)
    print(f"🎉 全部完成！")
    print(f"📥 总输入文件: {len(img_files)}")
    print(f"⏭️ 跳过厚层数据: {count_skipped_thick}")
    print(f"💾 成功处理并保存 (薄层): {count_processed}")
    print(f"📂 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()