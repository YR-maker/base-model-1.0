import os
import glob
import torch
import numpy as np
import monai
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensityRangePercentiles, EnsureType
)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ================= 1. 配置区域 (请根据实际情况修改) =================

# 权重路径 (你的 base.pt)
CKPT_PATH = "/home/yangrui/Project/Base-model/local_results/checkpoints/base.pt"

# 数据集根目录 (包含各个病例文件夹的父目录)
# 脚本会查找: DATA_ROOT/001/001.img.nii.gz
DATA_ROOT = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-All-Final/all"

# 输出结果保存目录
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-All-Final/select"

# 想要选取的样本数量 (例如 3-shot)
NUM_SELECTION = 10

# 统一缩放尺寸 (为了特征提取的一致性，建议缩放到网络训练时的 Patch Size)
INPUT_SIZE = (128, 256, 256)

# 设备
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"


# ================= 2. 模型定义 (基于 dyn_unet_base.yaml) =================

def get_model():
    # 参数来自你提供的 yaml 文件
    model_config = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": [
            [3, 3, 3], [3, 3, 3], [3, 3, 3],
            [3, 3, 3], [3, 3, 3], [3, 3, 3]
        ],
        "strides": [
            [1, 1, 1], [2, 2, 2], [2, 2, 2],
            [2, 2, 2], [2, 2, 2], [2, 2, 2]
        ],
        "upsample_kernel_size": [
            [2, 2, 2], [2, 2, 2], [2, 2, 2],
            [2, 2, 2], [2, 2, 2]
        ],
        "filters": [32, 64, 128, 256, 320, 320],
        "res_block": True
    }
    return DynUNet(**model_config)


def load_weights(model, path):
    """
    加载权重，自动处理 'model.' 或 'net.' 等前缀不匹配问题
    """
    print(f"Loading weights from {path}...")
    try:
        # map_location='cpu' 避免显存占用问题，加载后再转 GPU
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return model

    # 兼容处理：如果里面包了一层 'state_dict' 键 (Lightning checkponit)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # 创建新的 state_dict，移除不匹配的前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除常见前缀
        name = k.replace("model.", "").replace("net.", "").replace("model.", "")
        new_state_dict[name] = v

    # 尝试加载
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Success: Weights loaded strictly.")
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying strict=False...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Success with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model


# ================= 3. 特征提取器 =================

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = None
        self._register_hook()

    def _register_hook(self):
        """挂载 Hook 到 DynUNet 的 bottleneck 层"""

        def hook_fn(module, input, output):
            # output shape: [Batch, Channel, D, H, W]
            # 例如: [1, 320, 4, 4, 4]
            # 执行全局平均池化 (GAP) -> [Batch, Channel]
            self.features = torch.mean(output, dim=(2, 3, 4))

        # MONAI DynUNet 的最深层通常命名为 'bottleneck'
        if hasattr(self.model, 'bottleneck'):
            self.hook = self.model.bottleneck.register_forward_hook(hook_fn)
        else:
            raise AttributeError("Model does not have 'bottleneck' layer. Check architecture.")

    def extract(self, x):
        self.features = None
        self.model(x)  # 触发前向传播
        return self.features

    def close(self):
        if self.hook:
            self.hook.remove()


# ================= 4. 数据预处理与加载 =================

def get_transforms():
    """定义预处理流水线，确保与训练/推理时的数据分布一致"""
    return Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        # 强度归一化 (根据你的 configs/data/default_finetune.yaml)
        ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True),
        # 统一缩放到固定尺寸，以便提取特征
        Resize(spatial_size=INPUT_SIZE, mode="trilinear"),
        EnsureType(data_type="tensor")
    ])


def get_file_list(root_dir):
    """
    解析特定的目录结构: Dataset/ID/ID.img.nii.gz
    """
    file_list = []
    # 遍历 root_dir 下的所有子文件夹
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for subdir in subdirs:
        case_dir = os.path.join(root_dir, subdir)
        # 构建文件名：ID.img.nii.gz
        img_name = f"{subdir}.img.nii.gz"
        img_path = os.path.join(case_dir, img_name)

        if os.path.exists(img_path):
            file_list.append(img_path)
        else:
            # 兼容性查找：如果找不到标准命名，尝试找文件夹里的任意 .img.nii.gz
            candidates = glob.glob(os.path.join(case_dir, "*img.nii.gz"))
            if candidates:
                file_list.append(candidates[0])

    return sorted(file_list)


# ================= 5. 主流程 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- A. 准备模型 ---
    print("Initializing model...")
    model = get_model().to(DEVICE)
    model = load_weights(model, CKPT_PATH)
    model.eval()

    extractor = FeatureExtractor(model)
    transforms = get_transforms()

    # --- B. 准备数据 ---
    print(f"Scanning data in {DATA_ROOT} ...")
    file_paths = get_file_list(DATA_ROOT)
    print(f"Found {len(file_paths)} cases.")

    if len(file_paths) < NUM_SELECTION:
        print("Error: Candidate samples are fewer than target selection count.")
        return

    # --- C. 提取特征 ---
    features_list = []
    valid_paths = []

    print("Extracting features...")
    with torch.no_grad():
        for path in tqdm(file_paths):
            try:
                # 加载并预处理
                img_tensor = transforms(path)
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # [1, C, D, H, W]

                # 提取
                feat = extractor.extract(img_tensor)

                # 转 numpy
                features_list.append(feat.cpu().numpy().flatten())
                valid_paths.append(path)

            except Exception as e:
                print(f"Skipping {path}: {e}")

    # 转换为矩阵 [N, Feature_Dim]
    features_matrix = np.array(features_list)
    print(f"Feature matrix shape: {features_matrix.shape}")

    # 保存特征备份 (可选)
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), features_matrix)

    # --- D. 聚类与选择 (K-Means) ---
    print(f"Running K-Means (k={NUM_SELECTION})...")

    kmeans = KMeans(n_clusters=NUM_SELECTION, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_matrix)
    centers = kmeans.cluster_centers_

    # 计算每个样本到其所属聚类中心的距离
    # cdist 计算所有点到所有中心的距离，我们只取对应的
    dists = cdist(features_matrix, centers, metric='euclidean')

    selected_samples = []

    print("\n=== Selection Results ===")
    for i in range(NUM_SELECTION):
        # 找到属于第 i 个簇的所有样本索引
        cluster_indices = np.where(cluster_labels == i)[0]

        if len(cluster_indices) > 0:
            # 在该簇内，找到距离该簇中心最近的样本
            # dists[cluster_indices, i] 取出该簇样本到该中心的距离
            min_dist_idx_in_cluster = np.argmin(dists[cluster_indices, i])
            # 还原回全局索引
            global_idx = cluster_indices[min_dist_idx_in_cluster]
        else:
            # 兜底（极少情况）：如果簇为空，找全局离该中心最近的
            global_idx = np.argmin(dists[:, i])

        # 记录
        selected_path = valid_paths[global_idx]
        selected_samples.append(selected_path)

        # 打印日志
        case_id = os.path.basename(os.path.dirname(selected_path))  # 获取文件夹名作为ID
        print(f"Cluster {i + 1}: Case ID [{case_id}] (Dist: {dists[global_idx, i]:.4f}) -> {selected_path}")

    # --- E. 保存结果列表 ---
    save_txt_path = os.path.join(OUTPUT_DIR, f"selected_{NUM_SELECTION}shot.txt")
    with open(save_txt_path, "w") as f:
        for p in selected_samples:
            f.write(p + "\n")

    print(f"\nDone! Selected file paths saved to: {save_txt_path}")

    # 清理 Hook
    extractor.close()


if __name__ == "__main__":
    main()