import os
import shutil
from pathlib import Path
import re
import SimpleITK as sitk

# =================================================================
# ⚙️ 配置区域
# =================================================================

# 1. 输入路径
SRC_IMG_DIR = Path("/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/imagesTr")
SRC_LABEL_DIR = Path("/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/reannotated_fixed")

# 2. 输出路径
DST_ROOT_DIR = Path("/home/yangrui/Project/Base-model/datasets/MSD08/MSD-clip/all")

# 3. 裁剪参数
MARGIN = 0

# 4. 文件名匹配模式
SRC_IMG_PATTERN = "hepaticvessel_*.nii.gz"
SRC_LABEL_PREFIX = "hp"


# =================================================================

def extract_id(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def crop_and_save_force_rename(img_path, label_path, final_img_path, final_label_path, margin=0):
    print(f"   ⚡ 正在读取并裁剪...")
    try:
        image = sitk.ReadImage(str(img_path))
        label = sitk.ReadImage(str(label_path))

        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        binary_label = sitk.BinaryThreshold(label, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
        label_shape_filter.Execute(binary_label)

        if not label_shape_filter.HasLabel(1):
            print(f"   ⚠️ 警告：无前景，直接复制。")
            shutil.copy2(img_path, final_img_path)
            shutil.copy2(label_path, final_label_path)
            return False

        bbox = label_shape_filter.GetBoundingBox(1)
        x, y, z, w, h, d = bbox
        img_size = image.GetSize()

        new_x = max(0, x - margin)
        new_y = max(0, y - margin)
        new_z = max(0, z - margin)

        end_x = min(img_size[0], x + w + margin)
        end_y = min(img_size[1], y + h + margin)
        end_z = min(img_size[2], z + d + margin)

        new_w = end_x - new_x
        new_h = end_y - new_y
        new_d = end_z - new_z

        final_roi = [new_x, new_y, new_z, new_w, new_h, new_d]

        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetRegionOfInterest(final_roi)

        cropped_image = roi_filter.Execute(image)
        cropped_label = roi_filter.Execute(label)

        temp_img_name = final_img_path.parent / "temp_image_safe.nii.gz"
        temp_label_name = final_img_path.parent / "temp_label_safe.nii.gz"

        sitk.WriteImage(cropped_image, str(temp_img_name))
        sitk.WriteImage(cropped_label, str(temp_label_name))

        if temp_img_name.exists():
            shutil.move(str(temp_img_name), str(final_img_path))

        if temp_label_name.exists():
            shutil.move(str(temp_label_name), str(final_label_path))

        junk_hdr = final_img_path.parent / (final_img_path.name.replace(".nii.gz", ".hdr"))
        junk_img = final_img_path.parent / (final_img_path.name.replace(".nii.gz", ".img"))
        if junk_hdr.exists(): os.remove(junk_hdr)
        if junk_img.exists(): os.remove(junk_img)

        original_vol = img_size[0] * img_size[1] * img_size[2]
        new_vol = new_w * new_h * new_d
        ratio = (1 - new_vol / max(original_vol, 1)) * 100
        print(f"   ✂️ 裁剪完成: {img_size} -> {(new_w, new_h, new_d)}")
        print(f"   💾 体积减小: {ratio:.2f}%")
        return True

    except Exception as e:
        print(f"   ❌ 内部错误: {e}")
        return False


def main():
    if not SRC_IMG_DIR.exists() or not SRC_LABEL_DIR.exists():
        print(f"❌ 错误：源目录不存在！")
        return

    img_files = sorted(list(SRC_IMG_DIR.glob(SRC_IMG_PATTERN)))
    total_files = len(img_files)
    print(f"🔍 扫描到 {total_files} 个文件，开始基于原始ID处理...")
    print(f"📂 输出目录: {DST_ROOT_DIR}")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    # 【修改点 1】 不再使用 enumerate 生成的 index，只作为计数器显示进度
    for i, img_path in enumerate(img_files, start=1):
        original_name = img_path.name

        # 提取 ID 字符串 (例如 "007")
        case_id_str = extract_id(original_name)

        if not case_id_str:
            fail_count += 1
            print(f"[{i}/{total_files}] ❌ 无法从文件名提取 ID: {original_name}")
            continue

        # 【修改点 2】 将 "007" 转换为整数 7，再转回字符串 "7"
        # 这样文件夹就会是 "7" 而不是 "007"
        real_id = str(int(case_id_str))

        expected_label_name = f"{SRC_LABEL_PREFIX}{case_id_str}.nii.gz"  # 注意：源标签文件名通常还是带前导零的(hp007)，如果源标签是hp7，这里也需要改
        label_path = SRC_LABEL_DIR / expected_label_name

        if not label_path.exists():
            # 尝试一下不带前导零的匹配，以防万一
            label_path_alt = SRC_LABEL_DIR / f"{SRC_LABEL_PREFIX}{real_id}.nii.gz"
            if label_path_alt.exists():
                label_path = label_path_alt
            else:
                print(f"[{i}/{total_files}] ⚠️ 跳过：无标签 (ID: {real_id}, 原始: {case_id_str})")
                fail_count += 1
                continue

        # 【修改点 3】 使用 real_id ("7") 创建文件夹和文件名
        target_folder = DST_ROOT_DIR / real_id
        target_folder.mkdir(parents=True, exist_ok=True)

        # 最终目标文件名： 7.img.nii.gz
        target_img_path = target_folder / f"{real_id}.img.nii.gz"
        target_label_path = target_folder / f"{real_id}.label.nii.gz"

        print(f"[{i}/{total_files}] 处理 ID: {real_id} (原始文件: {original_name}) ...")

        if crop_and_save_force_rename(img_path, label_path, target_img_path, target_label_path, margin=MARGIN):
            success_count += 1
        else:
            fail_count += 1

    print("-" * 50)
    print(f"🎉 处理完成！")
    print(f"✅ 成功: {success_count}")
    print(f"📂 结果路径: {DST_ROOT_DIR}")


if __name__ == "__main__":
    main()