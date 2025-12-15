import os
import glob
import torch
import numpy as np
import monai
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensityRangePercentiles, EnsureType
)
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ================= 1. 配置区域 =================

# 权重路径
CKPT_PATH = "/home/yangrui/Project/Base-model/local_results/checkpoints/base.pt"

# 数据集根目录
# 假设结构为: /path/to/dataset/001/001.img.nii.gz
DATA_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-clip-0/all"

# 输出结果目录
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-clip-0/select"

# 你想选几个样本来做微调 (例如 3-shot)
NUM_SELECTION = 10

# 输入尺寸 (保持与训练一致)
INPUT_SIZE = (128, 256, 256)

# 设备
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


# ================= 2. 模型定义 =================

def get_model():
    """根据你的 dyn_unet_base.yaml 配置构建模型"""
    model_config = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "upsample_kernel_size": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "filters": [32, 64, 128, 256, 320, 320],
        "res_block": True
    }
    return DynUNet(**model_config)


def load_weights(model, path):
    """加载权重，自动处理前缀不匹配"""
    print(f"Loading weights from {path}...")
    try:
        # map_location='cpu' 防止加载时爆显存
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'model.', 'net.', 'model.' 等前缀
        name = k.replace("model.", "").replace("net.", "").replace("model.", "")
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Success: Weights loaded strictly.")
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying non-strict...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Success (Non-strict). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model


# ================= 3. 特征提取工具 =================

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = None
        self._register_hook()

    def _register_hook(self):
        """挂载 Hook 到 bottleneck 层"""
        if hasattr(self.model, 'bottleneck'):
            def hook_fn(module, input, output):
                # output: [B, C, D, H, W] -> Global Average Pooling -> [B, C]
                self.features = torch.mean(output, dim=(2, 3, 4))

            self.hook = self.model.bottleneck.register_forward_hook(hook_fn)
        else:
            raise AttributeError("Model missing 'bottleneck' layer!")

    def extract(self, x):
        self.features = None
        self.model(x)
        return self.features

    def close(self):
        if self.hook: self.hook.remove()


def get_transforms():
    """数据预处理"""
    return Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True),
        Resize(spatial_size=INPUT_SIZE, mode="trilinear"),
        EnsureType(data_type="tensor")
    ])


def get_all_files(root_dir):
    """获取所有 .img.nii.gz 文件并排序"""
    # 针对 Dataset/ID/ID.img.nii.gz 结构
    file_list = []
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for subdir in subdirs:
        case_dir = os.path.join(root_dir, subdir)
        # 优先匹配标准命名
        target_file = os.path.join(case_dir, f"{subdir}.img.nii.gz")
        if os.path.exists(target_file):
            file_list.append(target_file)
        else:
            # 备选匹配
            candidates = glob.glob(os.path.join(case_dir, "*img.nii.gz"))
            if candidates:
                file_list.append(sorted(candidates)[0])

    # 再次按文件名数字排序，确保顺序稳定
    try:
        file_list.sort(key=lambda x: int(os.path.basename(os.path.dirname(x))))
    except:
        file_list.sort()

    return file_list


# ================= 4. 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- A. 获取并划分数据 ---
    print(f"Scanning data in {DATA_ROOT}...")
    all_files = get_all_files(DATA_ROOT)
    total_count = len(all_files)

    if total_count == 0:
        print("Error: No files found!")
        return

    # 划分 90% Pool 和 10% Test
    split_index = int(total_count * 0.9)
    pool_files = all_files[:split_index]
    test_files = all_files[split_index:]

    print(f"Total files: {total_count}")
    print(f"Pool Size (90%): {len(pool_files)} (Training Candidates)")
    print(f"Test Size (10%): {len(test_files)} (Target Distribution)")
    print("-" * 30)
    print(f"Test IDs (Example): {[os.path.basename(os.path.dirname(f)) for f in test_files[:5]]} ...")
    print("-" * 30)

    # --- B. 初始化模型 ---
    model = get_model().to(DEVICE)
    model = load_weights(model, CKPT_PATH)
    model.eval()

    extractor = FeatureExtractor(model)
    transforms = get_transforms()

    # --- C. 提取特征 ---
    print("Extracting features for ALL data...")

    pool_features = []
    test_features = []

    # 提取函数
    def extract_features_list(file_list, desc):
        feats = []
        valid_files = []
        for path in tqdm(file_list, desc=desc):
            try:
                img_tensor = transforms(path).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = extractor.extract(img_tensor)
                feats.append(feat.cpu().numpy().flatten())
                valid_files.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
        return np.array(feats), valid_files

    # 分别提取
    pool_feats_matrix, valid_pool_files = extract_features_list(pool_files, "Processing Pool")
    test_feats_matrix, valid_test_files = extract_features_list(test_files, "Processing Test")

    # --- D. 核心逻辑：计算相似度与选择 ---
    print("\nCalculating Distribution Similarity...")

    # 1. 计算测试集的“平均脸” (Prototype / Mean Embedding)
    # 这代表了测试集的整体分布中心
    test_center = np.mean(test_feats_matrix, axis=0).reshape(1, -1)  # [1, 320]

    # 2. 计算 Pool 中所有样本到 Test Center 的欧式距离
    # dists shape: [N_pool, 1]
    dists = cdist(pool_feats_matrix, test_center, metric='euclidean').flatten()

    # 3. 排序：找距离最小的 Top-K
    # argsort 从小到大排序，返回索引
    sorted_indices = np.argsort(dists)
    top_k_indices = sorted_indices[:NUM_SELECTION]

    # --- E. 输出结果 ---
    print("\n" + "=" * 50)
    print(f"🚀 Top {NUM_SELECTION} Samples closest to Test Set Distribution:")
    print("=" * 50)

    selected_paths = []

    for rank, idx in enumerate(top_k_indices):
        file_path = valid_pool_files[idx]
        dist_val = dists[idx]
        case_id = os.path.basename(os.path.dirname(file_path))

        print(f"Rank {rank + 1}: Case [{case_id}] | Distance: {dist_val:.4f}")
        print(f"   Path: {file_path}")
        selected_paths.append(file_path)

    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, "selected_for_test_adaptation.txt")
    with open(save_path, "w") as f:
        f.write("\n".join(selected_paths))

    print("=" * 50)
    print(f"Selection saved to: {save_path}")

    extractor.close()


if __name__ == "__main__":
    main()