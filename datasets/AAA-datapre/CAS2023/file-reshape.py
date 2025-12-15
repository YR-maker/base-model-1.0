import os
import shutil
from pathlib import Path
from tqdm import tqdm  # 进度条库



def reformat_dataset():
    # ================= 配置路径 =================
    # 原始数据根目录
    origin_root = Path("/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-origin")
    img_dir = origin_root / "data"
    mask_dir = origin_root / "mask"

    # 目标输出目录
    target_root = Path("/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-reshape/all")

    # ===========================================

    # 1. 检查源目录是否存在
    if not img_dir.exists() or not mask_dir.exists():
        print(f"错误: 源目录不存在。\n请检查: {img_dir}\n或: {mask_dir}")
        return

    # 2. 创建目标根目录
    target_root.mkdir(parents=True, exist_ok=True)

    print(f"目标目录已创建/确认: {target_root}")

    # 3. 获取所有图像文件
    image_files = sorted(list(img_dir.glob("*.nii.gz")))
    print(f"找到 {len(image_files)} 个图像文件，开始处理...")

    success_count = 0
    error_count = 0

    # 使用 tqdm 显示进度条
    for img_path in tqdm(image_files, desc="重组数据结构"):
        file_name = img_path.name  # 例如: "000.nii.gz"

        # 对应的 mask 路径
        mask_path = mask_dir / file_name

        # 检查 Mask 是否存在
        if not mask_path.exists():
            print(f"\n[跳过] 未找到对应的 Mask 文件: {file_name}")
            error_count += 1
            continue

        # --- ID 转换逻辑 ---
        # 提取文件名主体 (去掉 .nii.gz)
        raw_id = file_name.replace(".nii.gz", "")  # 得到 "000"

        try:
            # 尝试转为整数再转字符串，实现 "000" -> "0"
            # 如果你的文件名包含非数字字符（如 "case_001"），这里会报错进入 except
            new_id = str(int(raw_id))
        except ValueError:
            # 如果文件名不是纯数字，则保持原样
            new_id = raw_id

        # --- 创建目标文件夹 ---
        # 目标路径: .../CAS2023-reshape/0/
        case_dir = target_root / new_id
        case_dir.mkdir(exist_ok=True)

        # --- 定义新文件名 ---
        # 0.img.nii.gz
        new_img_name = f"{new_id}.img.nii.gz"
        # 0.label.nii.gz
        new_mask_name = f"{new_id}.label.nii.gz"

        target_img_path = case_dir / new_img_name
        target_mask_path = case_dir / new_mask_name

        # --- 复制文件 ---
        try:
            # copy2 会保留文件的元数据（如创建时间）
            shutil.copy2(img_path, target_img_path)
            shutil.copy2(mask_path, target_mask_path)
            success_count += 1
        except Exception as e:
            print(f"\n[错误] 复制文件失败 {file_name}: {e}")
            error_count += 1

    print("-" * 30)
    print(f"处理完成!")
    print(f"成功: {success_count}")
    print(f"失败/跳过: {error_count}")
    print(f"数据已保存至: {target_root}")


if __name__ == "__main__":
    reformat_dataset()