import os
import numpy as np
import nibabel as nib
from tqdm import tqdm  # 用于显示进度条

# ================= 配置路径 =================
# 输入数据的根目录
BASE_DIR = "/home/yangrui/Project/Base-model/datasets/Hipas/hipas-origin"
IMG_DIR = os.path.join(BASE_DIR, "ct_scan")
ART_DIR = os.path.join(BASE_DIR, "annotation/artery")
VEIN_DIR = os.path.join(BASE_DIR, "annotation/vein")

# 输出保存的根目录 (你可以修改为你想要保存的地方)
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/Hipas/hipas-reshape"

# 数据集的数量
NUM_CASES = 250


# ================= 辅助函数 =================

def load_npz(file_path):
    """
    读取npz文件并返回其中的数组。
    这里假设npz中包含的第一个key就是数据（通常是'arr_0'或'data'）。
    """
    try:
        with np.load(file_path) as data:
            keys = list(data.keys())
            return data[keys[0]]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_nii(data, save_path, affine=np.eye(4)):
    """
    将numpy数组保存为nii.gz文件。
    如果没有提供仿射矩阵(affine)，默认使用单位矩阵。
    """
    # 确保数据是浮点型或整型，避免bool类型导致的兼容性问题
    if data.dtype == bool:
        data = data.astype(np.uint8)

    img = nib.Nifti1Image(data, affine)
    nib.save(img, save_path)


# ================= 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    print("开始转换数据...")

    # 遍历 001 到 250
    for i in tqdm(range(1, NUM_CASES + 1), desc="Processing"):
        case_id = f"{i:03d}"  # 格式化为 001, 002...

        # 定义输入文件路径
        img_path = os.path.join(IMG_DIR, f"{case_id}.npz")
        art_path = os.path.join(ART_DIR, f"{case_id}.npz")
        vein_path = os.path.join(VEIN_DIR, f"{case_id}.npz")

        # 检查文件是否存在
        if not (os.path.exists(img_path) and os.path.exists(art_path) and os.path.exists(vein_path)):
            print(f"\n警告: 跳过 {case_id}，因为缺少源文件。")
            continue

        # 1. 创建该编号的文件夹
        case_folder = os.path.join(OUTPUT_DIR, case_id)
        os.makedirs(case_folder, exist_ok=True)

        # 2. 读取数据
        img_data = load_npz(img_path)
        art_data = load_npz(art_path)
        vein_data = load_npz(vein_path)

        if img_data is None or art_data is None or vein_data is None:
            continue

        # 3. 保存原始影像 (数字.img.nii.gz)
        save_nii(img_data, os.path.join(case_folder, f"{case_id}.img.nii.gz"))

        # 4. 保存动脉标签 (数字.artery.nii.gz) -> 这里假设artery指动脉
        # 标签数据通常转为uint8以节省空间
        art_data = art_data.astype(np.uint8)
        save_nii(art_data, os.path.join(case_folder, f"{case_id}.artery.nii.gz"))

        # 5. 保存静脉标签 (数字.vein.nii.gz)
        vein_data = vein_data.astype(np.uint8)
        save_nii(vein_data, os.path.join(case_folder, f"{case_id}.vein.nii.gz"))

        # 6. 合并标签 (数字.label.nii.gz)
        # 逻辑：背景=0, 动脉=1, 静脉=2
        # 注意：如果同一体素既是动脉又是静脉（虽然解剖学上不应该），这里静脉会覆盖动脉
        combined_label = np.zeros_like(art_data, dtype=np.uint8)

        # 标记动脉为 1
        combined_label[art_data > 0] = 1
        # 标记静脉为 2
        combined_label[vein_data > 0] = 2

        save_nii(combined_label, os.path.join(case_folder, f"{case_id}.label.nii.gz"))

    print(f"\n转换完成！数据已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()