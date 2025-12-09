import os
import numpy as np
import nibabel as nib
from skimage import measure, morphology
from scipy import ndimage
import glob
import logging
from tqdm import tqdm



def get_body_mask(img_data):
    """
    逻辑源自你提供的 ROI.py
    """
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
    """
    逻辑源自你提供的 ROI.py
    """
    z_axis = 2  # 假设Z轴是第2维
    final_mask = np.zeros_like(lung_mask)

    for i in range(lung_mask.shape[z_axis]):
        if z_axis == 0:
            slice_img = lung_mask[i, :, :]
        elif z_axis == 1:
            slice_img = lung_mask[:, i, :]
        else:
            slice_img = lung_mask[:, :, i]

        if np.sum(slice_img) > 0:
            try:
                hull = morphology.convex_hull_image(slice_img)
                if z_axis == 0:
                    final_mask[i, :, :] = hull
                elif z_axis == 1:
                    final_mask[:, i, :] = hull
                else:
                    final_mask[:, :, i] = hull
            except Exception:
                pass
    return final_mask


def generate_roi_mask(img_data):
    """
    串联上述两个函数生成最终Mask
    """
    # 1. 初步提取
    lung_mask = get_body_mask(img_data)
    if np.sum(lung_mask) == 0:
        return None
    # 2. 填充心脏
    filled_mask = fill_hull_slice_by_slice(lung_mask)
    # 3. 腐蚀 (生成精确Mask)
    # 注意：这里我们生成Mask是为了计算包围盒，为了安全起见，
    # 腐蚀可以稍微少做一点，或者不做，以免把血管切出去。
    # 这里保持和你原来一样的逻辑，但在裁剪时我们会加 padding
    eroded_mask = ndimage.binary_erosion(filled_mask, iterations=5).astype(np.uint8)
    return eroded_mask


# ==========================================
# 2. 核心裁剪逻辑 (BBox Calculation)
# ==========================================

def get_bbox_from_mask(mask, margin=10):
    """
    根据 mask 计算包围盒，并添加安全边距 (margin)
    """
    # 找到所有非零元素的索引
    coords = np.argwhere(mask > 0)

    # 如果 mask 是空的
    if coords.size == 0:
        return None, None, None

    # 获取最小和最大坐标 (min_x, min_y, min_z)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1  # +1 因为切片是左闭右开

    # 添加 margin (安全边距)
    # 比如：如果心脏在 [100:200]，我们裁 [90:210]，防止边缘切断
    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, mask.shape)

    # 生成切片对象
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))

    return slices, min_coords, max_coords


def crop_and_save(img_path, lbl_path, output_dir):
    """
    读取图像和标签，生成Mask，应用Mask（去除背景），然后裁剪并保存
    """
    try:
        # 1. 读取图像
        img_obj = nib.load(img_path)
        img_data = img_obj.get_fdata()
        original_affine = img_obj.affine

        # 2. 生成 ROI Mask (全图尺寸)
        mask = generate_roi_mask(img_data)
        if mask is None:
            logging.warning(f"跳过: {img_path} 无法生成有效Mask")
            return

        # 3. 计算裁剪坐标 (BBox)
        # 向外扩充 margin=0
        crop_slices, min_coords, _ = get_bbox_from_mask(mask, margin=0)
        if crop_slices is None:
            logging.warning(f"跳过: {img_path} Mask为空")
            return

        # ==== 核心修改开始 ====

        # 4. 执行裁剪
        # 4.1 裁剪原图数据
        cropped_img_data = img_data[crop_slices].copy()  # 必须copy才能修改

        # 4.2 裁剪 Mask 数据 (保证和图是一一对应的)
        cropped_mask = mask[crop_slices]

        # 4.3 应用 Mask (这步才是真正的“遮罩”操作)
        # 获取图像背景最小值 (通常是 -1024 或 -3000)
        min_val = np.min(img_data)

        # 将 Mask 为 0 的区域（肋骨、背部等）全部置为背景值
        cropped_img_data[cropped_mask == 0] = min_val

        # ==== 核心修改结束 ====

        # 5. 更新 Affine (坐标原点变换)
        new_affine = original_affine.copy()
        translation = np.dot(original_affine[:3, :3], min_coords)
        new_affine[:3, 3] += translation

        # 6. 保存裁剪后的 Image (此时已经是去除了肋骨的干净心脏图)
        filename_img = os.path.basename(img_path)
        save_path_img = os.path.join(output_dir, filename_img)
        new_img_obj = nib.Nifti1Image(cropped_img_data, new_affine, img_obj.header)
        nib.save(new_img_obj, save_path_img)

        # 7. 处理 Label (如果存在)
        if lbl_path and os.path.exists(lbl_path):
            lbl_obj = nib.load(lbl_path)
            lbl_data = lbl_obj.get_fdata()

            # Label 不需要被Mask“涂黑”，只需要裁剪对齐即可
            # 因为 Label 只有 0和1，背景本来就是0
            cropped_lbl_data = lbl_data[crop_slices]

            # 保存 Label
            filename_lbl = os.path.basename(lbl_path)
            save_path_lbl = os.path.join(output_dir, filename_lbl)
            new_lbl_obj = nib.Nifti1Image(cropped_lbl_data.astype(np.uint8), new_affine, lbl_obj.header)
            nib.save(new_lbl_obj, save_path_lbl)

        logging.info(
            f"已处理: {os.path.basename(os.path.dirname(img_path))} -> 形状 {cropped_img_data.shape} (已去除背景)")

    except Exception as e:
        logging.error(f"处理失败 {img_path}: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
# 3. 主流程：遍历文件夹
# ==========================================

def main():
    # ==== 配置路径 ====
    # 输入：之前整理好的 Parse-formatted (或你当前的 Parse 文件夹)
    input_root = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-formatted"

    # 输出：新的 ROI 裁剪数据集
    output_root = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-ROI"

    splits = ["fine-tuning", "val", "test"]
    # ================

    for split in splits:
        src_split_path = os.path.join(input_root, split)
        dst_split_path = os.path.join(output_root, split)

        if not os.path.exists(src_split_path):
            continue

        # 遍历该 split 下的所有病人 ID 文件夹 (例如 "5", "10"...)
        patient_ids = [d for d in os.listdir(src_split_path) if os.path.isdir(os.path.join(src_split_path, d))]

        print(f"正在处理 {split} 集，共 {len(patient_ids)} 个病例...")

        for pid in tqdm(patient_ids):
            patient_src_dir = os.path.join(src_split_path, pid)
            patient_dst_dir = os.path.join(dst_split_path, pid)

            # 创建目标文件夹
            if not os.path.exists(patient_dst_dir):
                os.makedirs(patient_dst_dir)

            # 寻找 Image 和 Label
            # 兼容多种命名: image.nii.gz 或 5.img.nii.gz
            img_files = glob.glob(os.path.join(patient_src_dir, "*img.nii.gz")) + \
                        glob.glob(os.path.join(patient_src_dir, "image.nii.gz"))

            lbl_files = glob.glob(os.path.join(patient_src_dir, "*label.nii.gz")) + \
                        glob.glob(os.path.join(patient_src_dir, "label.nii.gz"))

            if not img_files:
                logging.warning(f"未找到图像文件: {patient_src_dir}")
                continue

            # 取第一个匹配到的文件
            img_path = img_files[0]
            lbl_path = lbl_files[0] if lbl_files else None

            # 执行裁剪
            crop_and_save(img_path, lbl_path, patient_dst_dir)

    print(f"\n全部完成！裁剪后的数据集保存在: {output_root}")


if __name__ == "__main__":
    main()