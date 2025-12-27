import os
import time
import numpy as np
import nibabel as nib
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

try:
    import cupy as cp

    HAS_GPU = True
    print("✅ GPU 加速环境 (CuPy) 检测正常")
except ImportError:
    raise RuntimeError("❌ 必须安装 cupy 才能使用此加速脚本！")


def build_skeleton_graph(skeleton):
    """构建骨架图 (CPU)"""
    z, y, x = np.where(skeleton > 0)
    nodes = list(zip(z, y, x))

    G = nx.Graph()
    for i, coord in enumerate(nodes):
        G.add_node(i, pos=coord)

    tree = cKDTree(nodes)
    pairs = tree.query_pairs(r=1.8)

    for i, j in pairs:
        G.add_edge(i, j, weight=1.0)

    return G, nodes


def find_root_guided_paths(G, nodes, image_shape, top_k=2):
    """
    基于解剖学位置寻找主干路径。
    """
    endpoints = [n for n, d in G.degree() if d == 1]
    if len(endpoints) < 2: return []

    # 定义“顶部中心”目标点 (Target Origin)
    all_z = [n[0] for n in nodes]
    min_z = np.min(all_z)
    target_origin = np.array([min_z, image_shape[1] / 2, image_shape[2] / 2])

    candidate_paths = []

    # 遍历每个连通分量
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        sub_endpoints = [n for n in endpoints if n in component]
        if not sub_endpoints: continue

        # A. 寻找该分量的 Root
        best_root = None
        min_dist = float('inf')

        for ep in sub_endpoints:
            coord = np.array(nodes[ep])
            dist = np.linalg.norm(coord - target_origin)
            if dist < min_dist:
                min_dist = dist
                best_root = ep

        # B. 从 Root 出发，找到最远的节点
        lengths = nx.single_source_shortest_path_length(subgraph, best_root)
        furthest_node = max(lengths, key=lengths.get)
        path = nx.shortest_path(subgraph, best_root, furthest_node)

        if len(path) > 20:
            candidate_paths.append(path)

    # 排序并取前 K 条
    candidate_paths.sort(key=len, reverse=True)

    top_paths_coords = []
    for i in range(min(top_k, len(candidate_paths))):
        path_nodes = candidate_paths[i]
        top_paths_coords.append([nodes[n] for n in path_nodes])

    return top_paths_coords


def reconstruct_trunk_sparse(mask_cpu, trunk_coords_list, dt_map_cpu, expansion_ratio):
    """
    GPU 稀疏重建：合并所有主干为 Label 1
    """
    print(f"🔥 Step 3: GPU 稀疏重建 (膨胀系数: {expansion_ratio})...")

    z_inds, y_inds, x_inds = np.where(mask_cpu > 0)
    vessel_coords_np = np.stack((z_inds, y_inds, x_inds), axis=1)

    vessel_coords_gpu = cp.asarray(vessel_coords_np, dtype=cp.float32)
    vessel_count = len(vessel_coords_np)

    vessel_labels_gpu = cp.zeros(vessel_count, dtype=cp.uint8)

    batch_size = 100000

    for idx, trunk_coords in enumerate(trunk_coords_list):
        label_id = 1  # 统统标记为 1 (主干)

        print(f"      -> 正在计算主干 {idx + 1} (并入 Label 1)...")

        trunk_coords_np = np.array(trunk_coords)
        trunk_radii_np = dt_map_cpu[trunk_coords_np[:, 0], trunk_coords_np[:, 1], trunk_coords_np[:, 2]]

        trunk_coords_gpu = cp.asarray(trunk_coords_np, dtype=cp.float32)
        trunk_radii_gpu = cp.asarray(trunk_radii_np, dtype=cp.float32) * expansion_ratio
        trunk_coords_broad = trunk_coords_gpu[None, :, :]

        for i in range(0, vessel_count, batch_size):
            end = min(i + batch_size, vessel_count)
            batch_vessel = vessel_coords_gpu[i:end][:, None, :]

            dists = cp.sqrt(cp.sum((batch_vessel - trunk_coords_broad) ** 2, axis=2))
            in_trunk = cp.any(dists <= trunk_radii_gpu[None, :], axis=1)

            current_labels = vessel_labels_gpu[i:end]
            vessel_labels_gpu[i:end] = cp.where(in_trunk, label_id, current_labels)

        del trunk_coords_gpu, trunk_radii_gpu, trunk_coords_broad
        cp.get_default_memory_pool().free_all_blocks()

    return vessel_labels_gpu, vessel_coords_np


def solve_coronary_root_guided(nii_path, expansion_ratio=2.0):
    """
    处理单个文件并覆盖保存
    """
    filename = os.path.basename(nii_path)
    print(f"\n🚀 正在处理: {nii_path}")
    print(f"   策略: 根部引导 + 二分类 (Trunk=1, Branch=2)")

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

    t0 = time.time()

    print("⏳ Step 1: 提取骨架 (CPU)...")
    try:
        skeleton = skeletonize(mask_cpu)
        dt_map_cpu = distance_transform_edt(mask_cpu)
    except Exception as e:
        print(f"❌ 骨架提取/距离变换失败: {e}")
        return

    print("🔍 Step 2: 搜索基于根部的主干路径...")
    G, all_nodes = build_skeleton_graph(skeleton)

    # 寻找前2条最长路径
    trunk_coords_list = find_root_guided_paths(G, all_nodes, mask_cpu.shape, top_k=2)

    if not trunk_coords_list:
        print("⚠️ 未找到有效主干，跳过")
        return

    # 重建 (内部所有主干都标为 1)
    try:
        vessel_labels_gpu, vessel_coords_np = reconstruct_trunk_sparse(mask_cpu, trunk_coords_list, dt_map_cpu,
                                                                       expansion_ratio)
    except Exception as e:
        print(f"❌ GPU重建失败: {e}")
        return

    print("🏷️  Step 4: 组装最终标签...")
    vessel_labels_cpu = cp.asnumpy(vessel_labels_gpu)
    del vessel_labels_gpu
    cp.get_default_memory_pool().free_all_blocks()

    result_label = np.zeros(mask_cpu.shape, dtype=np.uint8)
    z, y, x = vessel_coords_np[:, 0], vessel_coords_np[:, 1], vessel_coords_np[:, 2]
    result_label[z, y, x] = vessel_labels_cpu

    # 将分支 (mask存在 但 label为0) 设为 Label 2
    branch_mask = (mask_cpu > 0) & (result_label == 0)
    result_label[branch_mask] = 2

    print(f"💾 Step 5: 覆盖原文件: {nii_path} ...")

    # 覆盖保存
    new_img = nib.Nifti1Image(result_label, affine, header)
    nib.save(new_img, nii_path)

    trunk_pct = np.sum(result_label == 1) / np.sum(mask_cpu) * 100
    branch_pct = np.sum(result_label == 2) / np.sum(mask_cpu) * 100

    print(f"✅ 完成! 耗时: {time.time() - t0:.2f}s")
    print(f"   主干占比: {trunk_pct:.1f}%")
    print(f"   分支占比: {branch_pct:.1f}%")


if __name__ == "__main__":
    # ImageCAS 数据集根目录
    # 请根据实际情况修改这个路径
    dataset_root = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-ROI/"

    target_subdirs = ['train', 'val']

    # 膨胀系数 2.0 保证主干填充饱满
    ratio = 2.0

    print(f"开始批量处理目录: {dataset_root}")
    print(f"目标子目录: {target_subdirs}")
    print(f"膨胀系数: {ratio}")
    print("-" * 50)

    count = 0
    # 遍历目录
    for subdir in target_subdirs:
        search_path = os.path.join(dataset_root, subdir)
        if not os.path.exists(search_path):
            print(f"⚠️ 目录不存在: {search_path}")
            continue

        for root, dirs, files in os.walk(search_path):
            for file in files:
                # 匹配 label.nii.gz 文件
                if file.endswith("label.nii.gz"):
                    file_path = os.path.join(root, file)
                    try:
                        solve_coronary_root_guided(file_path, expansion_ratio=ratio)
                        count += 1
                    except Exception as e:
                        print(f"❌ 处理文件 {file_path} 时发生严重错误: {e}")

    print("-" * 50)
    print(f"🎉 全部结束! 共处理了 {count} 个文件。")