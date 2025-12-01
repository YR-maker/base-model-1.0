import os
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_closing, binary_opening,
    label, binary_fill_holes
)

def extract_heart_and_remove_bone(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder)
             if f.endswith(".nii") or f.endswith(".nii.gz")]

    print(f"[信息] 找到 {len(files)} 个待处理图像。")

    for idx, fname in enumerate(files):
        print(f"\n============ 处理 {idx+1}/{len(files)}: {fname} ============")

        input_path = os.path.join(input_folder, fname)

        # Step 1 ------------------------
        print("[1/10] 读取 CT 体数据...")
        nii = nib.load(input_path)
        img = nii.get_fdata().astype(np.float32)
        affine = nii.affine

        # Step 2 ------------------------
        print("[2/10] 心脏组织 HU 阈值筛选（含增强 CTA）...")
        heart_mask = (img > 20) & (img < 350)

        heart_mask = binary_closing(heart_mask, iterations=3)
        heart_mask = binary_fill_holes(heart_mask)

        # Step 3 ------------------------
        print("[3/10] 连通域：寻找最大心脏/主动脉区域...")
        labeled, num = label(heart_mask)
        if num == 0:
            print("  [警告] 无有效区域，跳过！")
            continue

        max_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        heart_mask = (labeled == max_label)

        # Step 4 ------------------------
        print("[4/10] 识别骨骼（高 HU 区域）...")
        bone_mask = img > 350

        # Step 5 ------------------------
        print("[5/10] 剔除与心脏连通的增强血管，避免误删主动脉...")
        bone_labeled, bone_num = label(bone_mask)

        for i in range(1, bone_num + 1):
            region = (bone_labeled == i)
            # 若区域与心脏接触，则保留（可能是主动脉/肺动脉）
            if np.any(region & heart_mask):
                bone_mask[region] = False  # 保留
            else:
                bone_mask[region] = True   # 这是骨骼

        # Step 6 ------------------------
        print("[6/10] 从最终 mask 中移除骨骼...")
        final_mask = heart_mask & (~bone_mask)

        # Step 7 ------------------------
        print("[7/10] 清理 mask 小碎片...")
        labeled_final, final_num = label(final_mask)
        # 保留最大体
        if final_num > 0:
            max_label_final = np.argmax(np.bincount(labeled_final.flat)[1:]) + 1
            final_mask = (labeled_final == max_label_final)

        # Step 8 ------------------------
        print("[8/10] 生成输出图像（其它区域全部设为 0）...")
        output_img = img * final_mask  # 0 即透明区域

        # Step 9 ------------------------
        print("[9/10] 保存结果 NIfTI...")
        out_nii = nib.Nifti1Image(output_img.astype(np.float32), affine)
        output_path = os.path.join(output_folder, fname)
        nib.save(out_nii, output_path)

        print(f"[完成] 已保存: {output_path}")

    print("\n全部处理完成！")


# ------------------------------
# 你只需修改这两个路径
# ------------------------------
input_folder = r"/home/yangrui/Project/Base-models/input/Parse/img"
output_folder = r"/home/yangrui/Project/Base-models/input/Parse/total"

extract_heart_and_remove_bone(input_folder, output_folder)
