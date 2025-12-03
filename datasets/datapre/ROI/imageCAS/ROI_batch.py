import os
import shutil
import time
import numpy as np
import nibabel as nib
from skimage import morphology, measure
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor  # 引入多进程加速

# ================= 配置区域 =================
# 输入数据的根目录
SRC_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-clip"

# 输出数据的根目录 (自动创建)
DST_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-ROI-200"

# 需要处理的子集文件夹
SUBSETS = ["train", "val", "test"]

# 你的成功参数
PARAM_THRESH = -200
PARAM_ERODE = 12
PARAM_DILATE = 20

# 是否复制 Label 文件 (建议开启，这样新数据集是完整的)
COPY_LABELS = True


# ===========================================

def extract_heart_advanced(input_path, output_path,
                           threshold_hu=-200,
                           erosion_radius=12,
                           dilation_radius=20):
    """
    核心处理算法 (基于你验证通过的逻辑)
    """
    try:
        # 1. 读取数据
        img = nib.load(input_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # 2. 基础阈值
        binary_mask = data > threshold_hu

        # 3. 核心分离 (腐蚀)
        selem_erode = morphology.ball(erosion_radius)
        eroded_mask = morphology.binary_erosion(binary_mask, selem_erode)

        # 4. 提取最大连通域
        labeled_mask, num_features = ndimage.label(eroded_mask)
        if num_features == 0:
            print(f"[Warning] {os.path.basename(input_path)}: 腐蚀后无残留，跳过处理，直接复制原图。")
            shutil.copy(input_path, output_path)  # 保底措施
            return

        region_sizes = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
        region_sizes.sort(key=lambda x: x[1], reverse=True)
        largest_label = region_sizes[0][0]
        heart_core = (labeled_mask == largest_label)

        # 5. 建立安全区 (距离变换)
        dist_map = ndimage.distance_transform_edt(~heart_core)
        safety_zone = dist_map <= dilation_radius

        # 6. 最终切割
        final_mask = np.logical_and(binary_mask, safety_zone)
        final_mask = morphology.binary_closing(final_mask, morphology.ball(2))

        # 7. 应用遮罩 (-1024)
        masked_data = data.copy()
        masked_data[~final_mask] = -200

        # 8. 保存
        new_img = nib.Nifti1Image(masked_data, affine, header)
        nib.save(new_img, output_path)
        return True

    except Exception as e:
        print(f"[Error] 处理 {input_path} 失败: {e}")
        return False


def process_single_case(args):
    """
    单个病例的处理逻辑 (用于多进程调用)
    """
    case_path, dst_subset_dir, subset_name = args

    case_name = os.path.basename(case_path)  # e.g., "1"

    # 构建源文件路径
    img_name = f"{case_name}.img.nii.gz"
    label_name = f"{case_name}.label.nii.gz"

    src_img_path = os.path.join(case_path, img_name)
    src_label_path = os.path.join(case_path, label_name)

    # 构建目标文件夹路径
    dst_case_dir = os.path.join(dst_subset_dir, case_name)
    os.makedirs(dst_case_dir, exist_ok=True)

    dst_img_path = os.path.join(dst_case_dir, img_name)
    dst_label_path = os.path.join(dst_case_dir, label_name)

    # 1. 处理图像
    if os.path.exists(src_img_path):
        # 检查是否已经存在，如果存在跳过 (方便断点续传)
        if not os.path.exists(dst_img_path):
            print(f"处理: {subset_name}/{case_name} ...")
            extract_heart_advanced(src_img_path, dst_img_path,
                                   threshold_hu=PARAM_THRESH,
                                   erosion_radius=PARAM_ERODE,
                                   dilation_radius=PARAM_DILATE)
        else:
            print(f"跳过 (已存在): {subset_name}/{case_name}")
    else:
        print(f"[Missing] 找不到图像: {src_img_path}")

    # 2. 复制标签 (如果开启)
    if COPY_LABELS and os.path.exists(src_label_path):
        if not os.path.exists(dst_label_path):
            shutil.copy(src_label_path, dst_label_path)


def main():
    print("=== 开始批量心脏ROI提取 ===")
    print(f"源目录: {SRC_ROOT}")
    print(f"目标目录: {DST_ROOT}")
    print(f"参数: Thresh={PARAM_THRESH}, Erode={PARAM_ERODE}, Dilate={PARAM_DILATE}")

    tasks = []

    # 遍历目录结构收集任务
    for subset in SUBSETS:
        src_subset_path = os.path.join(SRC_ROOT, subset)
        dst_subset_path = os.path.join(DST_ROOT, subset)

        if not os.path.exists(src_subset_path):
            print(f"警告: 目录不存在 {src_subset_path}")
            continue

        # 获取该子集下的所有病例文件夹 (数字命名的文件夹)
        case_folders = [f for f in os.listdir(src_subset_path) if os.path.isdir(os.path.join(src_subset_path, f))]

        # 按数字排序，看起来舒服点
        try:
            case_folders.sort(key=lambda x: int(x))
        except:
            case_folders.sort()

        for case_name in case_folders:
            case_path = os.path.join(src_subset_path, case_name)
            tasks.append((case_path, dst_subset_path, subset))

    print(f"\n共发现 {len(tasks)} 个病例，准备开始处理...")

    # 使用多进程并行处理 (根据CPU核数自动调度)，极大加快速度
    # 如果不想用多进程，可以把 max_workers 设为 1
    start_global = time.time()

    # max_workers 可以根据你的CPU核数调整，通常设为 4-8 比较合适
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_single_case, tasks)

    print("\n" + "=" * 30)
    print(f"全部完成！总耗时: {time.time() - start_global:.2f} 秒")
    print(f"新数据集位置: {DST_ROOT}")


if __name__ == "__main__":
    main()