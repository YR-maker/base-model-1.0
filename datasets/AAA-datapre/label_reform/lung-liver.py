import os
import time
import numpy as np
import nibabel as nib

# 必须有 CuPy
try:
    import cupy as cp
    from cupyx.scipy.ndimage import binary_erosion, binary_dilation
    from cupyx.scipy.ndimage import label as gpu_label
    from cupyx.scipy.ndimage import distance_transform_edt as gpu_edt

    print("✅ GPU 环境 (CuPy) 检测正常")
except ImportError:
    raise RuntimeError("❌ 必须安装 cupy 才能使用此加速脚本！")


def generate_ball(radius):
    """生成球形结构元素"""
    r = int(np.ceil(radius))
    z, y, x = cp.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    mask = (x ** 2 + y ** 2 + z ** 2) <= radius ** 2
    return mask


def solve_vessel_erosion_recovery(nii_path, thickness_ratio=0.5):
    """
    处理单个文件并直接覆盖保存
    """
    filename = os.path.basename(nii_path)
    print(f"\n🚀 正在处理: {nii_path}")
    print(f"   策略: 物理腐蚀-恢复法 (Erosion-Recovery)")

    # 1. 读取数据
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    mask_cpu = (data > 0).astype(np.uint8)
    if np.sum(mask_cpu) == 0:
        print("❌ Mask为空，跳过")
        return

    # 检查是否已经是处理过的文件 (可选：防止重复处理)
    # 如果已经是1和2的标签，并且不再全是1，可能已经处理过，这里不做硬性限制，直接覆盖

    t0 = time.time()

    # 2. 上传 GPU
    print("📤 Step 1: 上传 GPU 并分析最大厚度...")
    mask_gpu = cp.asarray(mask_cpu, dtype=bool)

    # 计算全图距离场
    dt_gpu = gpu_edt(mask_gpu)
    max_radius = float(cp.max(dt_gpu))

    # 设定腐蚀半径
    erode_radius = max_radius * thickness_ratio

    # 限制最小腐蚀半径
    if erode_radius < 2.0: erode_radius = 2.0

    print(f"   -> 检测到最大半径: {max_radius:.1f} px")
    print(f"   -> 设定腐蚀半径 R: {erode_radius:.1f} px")

    # -----------------------------------------------------------
    # Step 2: 强力腐蚀 (剥离分支)
    # -----------------------------------------------------------
    print(f"🔪 Step 2: 执行强力腐蚀...")

    struct_erode = generate_ball(erode_radius)
    eroded_mask = binary_erosion(mask_gpu, structure=struct_erode)

    if cp.sum(eroded_mask) == 0:
        print("⚠️ 警告: 腐蚀后图像为空！说明该样本血管太细，无法提取主干。保留原始标签。")
        return

    # -----------------------------------------------------------
    # Step 3: 只保留最大的连通域
    # -----------------------------------------------------------
    print(f"🔍 Step 3: 提取最大连通域...")

    labeled_array, num_features = gpu_label(eroded_mask)
    counts = cp.bincount(labeled_array.ravel())
    if len(counts) > 1:
        largest_label = cp.argmax(counts[1:]) + 1
        core_trunk_mask = (labeled_array == largest_label)
    else:
        core_trunk_mask = eroded_mask

    del labeled_array, counts, eroded_mask

    # -----------------------------------------------------------
    # Step 4: 过度膨胀 (恢复主干)
    # -----------------------------------------------------------
    print(f"🎈 Step 4: 过度膨胀回填 (R + 3.0 px)...")

    dilate_radius = erode_radius + 3.0
    struct_dilate = generate_ball(dilate_radius)
    restored_trunk = binary_dilation(core_trunk_mask, structure=struct_dilate)
    final_trunk = restored_trunk & mask_gpu

    # -----------------------------------------------------------
    # Step 5: 生成标签与覆盖保存
    # -----------------------------------------------------------
    print("🏷️  Step 5: 生成最终标签...")

    result_gpu = cp.zeros_like(mask_gpu, dtype=cp.uint8)
    result_gpu[final_trunk] = 1  # 主干
    result_gpu[mask_gpu & (~final_trunk)] = 2  # 分支

    # 统计
    trunk_ratio = cp.sum(final_trunk) / cp.sum(mask_gpu) * 100
    print(f"   -> 主干体积占比: {trunk_ratio:.2f}%")

    # 下载
    result_cpu = cp.asnumpy(result_gpu)

    # 释放显存
    del mask_gpu, dt_gpu, final_trunk, restored_trunk, core_trunk_mask, result_gpu
    cp.get_default_memory_pool().free_all_blocks()

    print(f"💾 Step 6: 覆盖原文件: {nii_path} ...")

    # 直接覆盖保存
    new_img = nib.Nifti1Image(result_cpu, affine, header)
    nib.save(new_img, nii_path)

    print(f"✅ 处理完成! 耗时: {time.time() - t0:.2f}s\n")


if __name__ == "__main__":
    # 数据集根目录
    dataset_root = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-61/"

    # 需要处理的子目录名称
    target_subdirs = ['train', 'val']

    # 肺部血管建议参数

    ratio = 0.3

    print(f"开始批量处理目录: {dataset_root}")
    print(f"目标子目录: {target_subdirs}")
    print(f"腐蚀比例: {ratio}")
    print("-" * 50)

    count = 0
    # 遍历目录
    for subdir in target_subdirs:
        search_path = os.path.join(dataset_root, subdir)
        if not os.path.exists(search_path):
            print(f"⚠️ 目录不存在: {search_path}")
            continue

        # os.walk 递归遍历所有子文件夹
        for root, dirs, files in os.walk(search_path):
            for file in files:
                # 匹配文件名
                if file.endswith("label.nii.gz"):
                    file_path = os.path.join(root, file)
                    try:
                        solve_vessel_erosion_recovery(file_path, thickness_ratio=ratio)
                        count += 1
                    except Exception as e:
                        print(f"❌ 处理文件 {file_path} 时发生严重错误: {e}")

    print("-" * 50)
    print(f"🎉 全部结束! 共处理了 {count} 个文件。")