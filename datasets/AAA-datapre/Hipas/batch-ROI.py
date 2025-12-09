import os
import glob
import numpy as np
import nibabel as nib
from skimage import measure, morphology
from scipy import ndimage
from tqdm import tqdm
import logging

# ================= 配置路径 =================
# 输入：上一各步骤生成的 NIfTI 数据集路径
INPUT_ROOT = "/home/yangrui/Project/Base-model/datasets/Hipas/hipas-reshape"
# 输出：处理后的 ROI 数据集路径
OUTPUT_ROOT = "/home/yangrui/Project/Base-model/datasets/Hipas/hipas-ROI/test"


# ================= 1. 核心 Mask 生成逻辑 (复用 batch-ROI.py) =================

def get_body_mask(img_data):
    """提取人体/肺部区域"""
    # 阈值分割
    binary = img_data < -300
    labels = measure.label(binary, connectivity=1)

    # 去除背景
    d, h, w = img_data.shape
    corners = [
        (0, 0, 0), (d - 1, 0, 0), (0, h - 1, 0), (d - 1, h - 1, 0),
        (0, 0, w - 1), (d - 1, 0, w - 1), (0, h - 1, w - 1), (d - 1, h - 1, w - 1)
    ]
    background_labels = set()
    for c in corners:
        bg_lbl = labels[c]
        if bg_lbl != 0:
            background_labels.add(bg_lbl)
    for bg_lbl in background_labels:
        labels[labels == bg_lbl] = 0

    # 保留最大连通域
    props = measure.regionprops(labels)
    props.sort(key=lambda x: x.area, reverse=True)

    lung_mask = np.zeros_like(binary, dtype=np.uint8)
    if not props:
        return lung_mask

    lung_mask[labels == props[0].label] = 1
    if len(props) > 1:
        if props[1].area > (props[0].area * 0.2):
            lung_mask[labels == props[1].label] = 1
    return lung_mask


def fill_hull_slice_by_slice(lung_mask):
    """逐层填充凸包"""
    z_axis = 2
    final_mask = np.zeros_like(lung_mask)

    for i in range(lung_mask.shape[z_axis]):
        # 兼容不同轴向，通常医学图像是 (D, H, W) 或 (H, W, D)
        # 这里为了稳健，简单判定一下形状，如果出错则跳过
        try:
            slice_img = lung_mask[:, :, i]  # 假设最后维度是Z
            if np.sum(slice_img) > 0:
                hull = morphology.convex_hull_image(slice_img)
                final_mask[:, :, i] = hull
        except Exception:
            pass
    return final_mask


def generate_roi_mask(img_data):
    """生成最终的心脏 ROI Mask"""
    # 1. 初步提取
    lung_mask = get_body_mask(img_data)
    if np.sum(lung_mask) == 0:
        return None
    # 2. 填充心脏
    filled_mask = fill_hull_slice_by_slice(lung_mask)
    # 3. 腐蚀 (生成精确Mask)
    eroded_mask = ndimage.binary_erosion(filled_mask, iterations=5).astype(np.uint8)
    return eroded_mask


# ================= 2. 裁剪与坐标计算逻辑 =================

def get_bbox_from_mask(mask, margin=0):
    """计算包围盒"""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None, None, None

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, mask.shape)

    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return slices, min_coords, max_coords


def save_nii(data, affine, path):
    """保存 NIfTI 文件辅助函数"""
    new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, path)


# ================= 3. 主处理流程 =================

def process_case(case_id, case_input_dir, case_output_dir):
    # 定义四个文件的文件名
    files = {
        "img": f"{case_id}.img.nii.gz",
        "artery": f"{case_id}.artery.nii.gz",
        "vein": f"{case_id}.vein.nii.gz",
        "label": f"{case_id}.label.nii.gz"
    }

    img_path = os.path.join(case_input_dir, files["img"])
    if not os.path.exists(img_path):
        print(f"Skipping {case_id}: Image not found.")
        return

    try:
        # --- 1. 处理主图像 ---
        img_obj = nib.load(img_path)
        img_data = img_obj.get_fdata()
        original_affine = img_obj.affine

        # 生成 Mask
        roi_mask = generate_roi_mask(img_data)
        if roi_mask is None:
            print(f"Skipping {case_id}: Failed to generate mask.")
            return

        # 计算裁剪范围 (BBox)
        crop_slices, min_coords, _ = get_bbox_from_mask(roi_mask, margin=0)
        if crop_slices is None:
            print(f"Skipping {case_id}: Empty mask.")
            return

        # 裁剪图像
        cropped_img = img_data[crop_slices].copy()

        # 应用遮罩 (将肋骨/背景设为最小值)
        cropped_mask = roi_mask[crop_slices]
        min_val = np.min(img_data)  # 获取背景值 (如 -3024)
        cropped_img[cropped_mask == 0] = min_val

        # 更新坐标原点 (Affine)
        new_affine = original_affine.copy()
        translation = np.dot(original_affine[:3, :3], min_coords)
        new_affine[:3, 3] += translation

        # 保存处理后的图像
        os.makedirs(case_output_dir, exist_ok=True)
        save_nii(cropped_img, new_affine, os.path.join(case_output_dir, files["img"]))

        # --- 2. 处理三个标签文件 ---
        # 标签只需要裁剪，不需要把背景值设为 -3024，否则标签值会乱
        label_keys = ["artery", "vein", "label"]

        for key in label_keys:
            lbl_path = os.path.join(case_input_dir, files[key])
            if os.path.exists(lbl_path):
                lbl_obj = nib.load(lbl_path)
                lbl_data = lbl_obj.get_fdata()

                # 仅裁剪
                cropped_lbl = lbl_data[crop_slices]

                # 确保是整数类型保存
                cropped_lbl = cropped_lbl.astype(np.uint8)

                save_nii(cropped_lbl, new_affine, os.path.join(case_output_dir, files[key]))

    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        import traceback
        traceback.print_exc()


def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"Input directory not found: {INPUT_ROOT}")
        return

    # 获取所有病例文件夹 (001, 002...)
    case_dirs = sorted([d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))])

    print(f"Found {len(case_dirs)} cases. Starting processing...")

    for case_id in tqdm(case_dirs):
        case_in = os.path.join(INPUT_ROOT, case_id)
        case_out = os.path.join(OUTPUT_ROOT, case_id)

        process_case(case_id, case_in, case_out)

    print(f"\nProcessing complete. Data saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()