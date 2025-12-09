import os
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm


def process_dataset():
    # ==== 配置路径 ====
    # 原始数据根目录
    source_root = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-origin"

    # 新的目标目录 (建议新建一个，不要覆盖原有的)
    target_root = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-formatted"

    # 需要处理的三个划分
    splits = ["fine-tuning", "val", "test"]
    # ================

    if not os.path.exists(target_root):
        os.makedirs(target_root)
        print(f"创建目标根目录: {target_root}")

    for split in splits:
        src_split_path = os.path.join(source_root, split)
        dst_split_path = os.path.join(target_root, split)

        if not os.path.exists(src_split_path):
            print(f"跳过: 找不到源文件夹 {src_split_path}")
            continue

        # 获取该 split 下的所有病人文件夹 (如 PA000005)
        patient_folders = [f for f in os.listdir(src_split_path) if os.path.isdir(os.path.join(src_split_path, f))]

        print(f"正在处理 {split} 集，共 {len(patient_folders)} 个病例...")

        for patient_folder in tqdm(patient_folders):
            # 1. 解析 ID: PA000005 -> 5
            try:
                # 去掉 'PA' 并转为 int 去掉前导零，再转回 str
                # 例如: PA000005 -> 5 -> "5"
                patient_id_raw = patient_folder.replace("PA", "")
                patient_id = str(int(patient_id_raw))
            except ValueError:
                print(f"警告: 无法解析文件夹名 {patient_folder}，将直接使用原名")
                patient_id = patient_folder

            # 2. 创建目标文件夹: .../test/5
            target_patient_dir = os.path.join(dst_split_path, patient_id)
            if not os.path.exists(target_patient_dir):
                os.makedirs(target_patient_dir)

            # 源文件路径
            src_img_path = os.path.join(src_split_path, patient_folder, "image.nii.gz")
            src_lbl_path = os.path.join(src_split_path, patient_folder, "label.nii")

            # 有时候 label 可能是 .nii.gz，做个兼容检查
            if not os.path.exists(src_lbl_path):
                src_lbl_path_gz = os.path.join(src_split_path, patient_folder, "label.nii.gz")
                if os.path.exists(src_lbl_path_gz):
                    src_lbl_path = src_lbl_path_gz

            # 目标文件路径
            # image.nii.gz -> 5.img.nii.gz
            dst_img_name = f"{patient_id}.img.nii.gz"
            dst_img_path = os.path.join(target_patient_dir, dst_img_name)

            # label.nii -> 5.label.nii.gz
            dst_lbl_name = f"{patient_id}.label.nii.gz"
            dst_lbl_path = os.path.join(target_patient_dir, dst_lbl_name)

            # 3. 处理 Image (直接复制，因为都是 .nii.gz)
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"缺失: {patient_folder} 中没有 image.nii.gz")

            # 4. 处理 Label (可能需要格式转换 .nii -> .nii.gz)
            if os.path.exists(src_lbl_path):
                # 如果源文件已经是 .nii.gz，直接复制
                if src_lbl_path.endswith(".nii.gz"):
                    shutil.copy2(src_lbl_path, dst_lbl_path)
                # 如果源文件是 .nii，使用 nibabel 读取并保存为压缩格式
                elif src_lbl_path.endswith(".nii"):
                    try:
                        lbl_obj = nib.load(src_lbl_path)
                        # 保存时文件名以 .nii.gz 结尾，nibabel 会自动压缩
                        nib.save(lbl_obj, dst_lbl_path)
                    except Exception as e:
                        print(f"转换出错: {src_lbl_path} -> {e}")
            else:
                print(f"缺失: {patient_folder} 中没有 label 文件")

    print("\n所有处理完成！")
    print(f"新数据集位于: {target_root}")


if __name__ == "__main__":
    process_dataset()