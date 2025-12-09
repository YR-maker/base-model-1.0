"""
多GPU模型推理脚本
功能: 自动将数据集切分到多个GPU上并行推理，最后汇总指标。
"""
import logging
import warnings
from pathlib import Path
import math
import os

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import hydra
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# 保持原有的引用不变
from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics

warnings.filterwarnings("ignore")
# 配置多进程启动方式，CUDA必须使用 spawn
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)


def load_model(cfg, device):
    """加载模型权重"""
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    try:
        # map_location 确保加载到正确的 GPU
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")

    model = hydra.utils.instantiate(cfg.model)
    state_dict_to_load = None

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            key = key.replace('model.', '').replace('models.', '').replace('net.', '').replace('module.', '')
            new_state_dict[key] = value
        state_dict_to_load = new_state_dict
    elif isinstance(ckpt, dict) and any(key.startswith(('encoder', 'decoder', 'backbone')) for key in ckpt.keys()):
        state_dict_to_load = ckpt
    else:
        state_dict_to_load = ckpt

    if state_dict_to_load is not None:
        try:
            model.load_state_dict(state_dict_to_load, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict_to_load, strict=False)
    else:
        raise ValueError("权重文件格式不支持")

    return model


def get_paths_nested(cfg):
    """获取所有文件路径"""
    root_dir = Path(hydra.utils.to_absolute_path(cfg.image_path))
    if not root_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")

    image_paths = []
    mask_paths = []

    case_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    logger.info(f"在 {root_dir} 中找到 {len(case_dirs)} 个子文件夹")

    for case_dir in case_dirs:
        case_id = case_dir.name
        img_name = f"{case_id}.img.nii.gz"
        img_p = case_dir / img_name

        if img_p.exists():
            image_paths.append(img_p)
            if cfg.get('mask_suffix') and cfg.get('mask_path'):
                label_name = f"{case_id}{cfg.mask_suffix}"
                label_p = case_dir / label_name
                if label_p.exists():
                    mask_paths.append(label_p)
                elif cfg.get('strict_matching', False):
                    logger.warning(f"Case {case_id}: 未找到标签文件 {label_name}")

    if not image_paths:
        raise FileNotFoundError(f"未找到符合规则的图像。")

    if mask_paths and len(mask_paths) != len(image_paths):
        if cfg.get('strict_matching', True):
            raise ValueError("标签数量不匹配")
        else:
            mask_paths = None

    if not mask_paths:
        return image_paths, None

    return image_paths, mask_paths


def resample(image, factor=None, target_shape=None):
    if factor == 1: return image
    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


# --- 后处理辅助函数 ---
def keep_largest_vessels(prediction, num_vessels=3):
    if isinstance(prediction, torch.Tensor): prediction = prediction.cpu().numpy()
    prediction = prediction.astype(int)
    labeled_mask = label(prediction, connectivity=3)
    regions = regionprops(labeled_mask)
    if len(regions) <= num_vessels: return prediction
    regions_sorted = sorted(regions, key=lambda x: x.area, reverse=True)
    processed_mask = np.zeros_like(prediction)
    for i in range(min(num_vessels, len(regions_sorted))):
        coords = regions_sorted[i].coords
        processed_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return processed_mask

def keep_closest_vessels(prediction, num_vessels=3):
    if isinstance(prediction, torch.Tensor): prediction = prediction.cpu().numpy()
    prediction = prediction.astype(int)
    labeled_mask = label(prediction, connectivity=3)
    regions = regionprops(labeled_mask)
    if len(regions) <= num_vessels: return prediction
    image_center = np.array(prediction.shape) / 2.0
    regions_sorted = sorted(regions, key=lambda x: np.linalg.norm(np.array(x.centroid) - image_center))
    processed_mask = np.zeros_like(prediction)
    for i in range(min(num_vessels, len(regions_sorted))):
        coords = regions_sorted[i].coords
        processed_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return processed_mask


def run_inference_worker(rank, gpu_id, image_paths, mask_paths, cfg, return_dict):
    """
    单个 Worker 进程：负责在一个 GPU 上跑一部分数据
    """
    device = torch.device(f"cuda:{gpu_id}")

    # 设置随机种子
    np.random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)

    # 加载模型
    try:
        model = load_model(cfg, device)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {e}")
        return

    transforms = generate_transforms(cfg.transforms_config)

    # 确定 I/O
    first_name = image_paths[0].name
    if 'nii.gz' in first_name:
        rw_suffix = 'nii.gz'
        save_ext = '.nii.gz'
    else:
        rw_suffix = image_paths[0].suffix
        save_ext = image_paths[0].suffix

    image_reader_writer = determine_reader_writer(rw_suffix)()
    save_writer = determine_reader_writer(rw_suffix)()

    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.batch_size,
        overlap=cfg.overlap,
        mode=cfg.mode,
        sigma_scale=cfg.sigma_scale,
        padding_mode=cfg.padding_mode
    )

    local_metrics = {}

    # 输出设置
    save_predictions = False
    output_folder = None
    if cfg.output_folder and str(cfg.output_folder).lower() != "none":
        save_predictions = True
        output_folder = Path(cfg.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # 进度条
    desc = f"GPU {gpu_id}"
    # 使用 position 让进度条不重叠，rank 0 在最上面
    iterator = tqdm(enumerate(image_paths), total=len(image_paths), desc=desc, position=rank, leave=True)

    with torch.no_grad():
        for idx, image_path in iterator:
            preds = []
            try:
                img_data = image_reader_writer.read_images(image_path)[0].astype(np.float32)
            except Exception as e:
                print(f"[GPU {gpu_id}] 读取失败 {image_path}: {e}")
                continue

            # --- TTA & 推理 ---
            for scale in cfg.tta.scales:
                image_tensor = transforms(img_data)
                if isinstance(image_tensor, np.ndarray):
                    image_tensor = torch.from_numpy(image_tensor)
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                image = image_tensor.unsqueeze(0).to(device)

                if cfg.tta.invert and image.mean() > cfg.tta.invert_mean_thresh:
                    image = 1 - image

                original_shape = image.shape
                image = resample(image, factor=scale)

                logits = inferer(image, model)
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.cpu().squeeze())

            # --- 融合 ---
            if len(preds) > 1:
                stacked_preds = torch.stack(preds)
                if cfg.merging.max:
                    pred = stacked_preds.max(dim=0)[0].sigmoid()
                else:
                    pred = stacked_preds.mean(dim=0).sigmoid()
            else:
                pred = preds[0].sigmoid()

            pred_thresh = (pred > cfg.merging.threshold).numpy()

            # --- 后处理 ---
            if cfg.post.apply:
                pred_thresh = remove_small_objects(
                    pred_thresh.astype(bool),
                    min_size=cfg.post.small_objects_min_size,
                    connectivity=cfg.post.small_objects_connectivity
                )
            if cfg.post.get('keep_largest_vessels', False):
                pred_thresh = keep_largest_vessels(pred_thresh.astype(int), cfg.post.num_largest_vessels)
            if cfg.post.get('keep_closest_vessels', False):
                pred_thresh = keep_closest_vessels(pred_thresh.astype(int), cfg.post.num_closest_vessels)

            # --- 保存 ---
            if save_predictions:
                clean_name = image_path.name.replace('.img.nii.gz', '').replace('.nii.gz', '')
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
                save_path = output_folder / save_name
                save_writer.write_seg(pred_thresh.astype(np.uint8), save_path)

            # --- 计算指标 & 打印 (这是你需要的) ---
            if mask_paths:
                # 找到对应的 mask 路径
                # 注意：因为切分了数据，image_paths 和 mask_paths 在这个进程里是一一对应的
                # 但为了安全，我们用 image_reader_writer 再读一次，或者如果你传进来的 mask_paths 已经是切分好的列表
                # 这里假设 mask_paths 是和 image_paths 对齐的切片列表
                if mask_paths[idx] is not None:
                    mask_data = image_reader_writer.read_images(mask_paths[idx])[0]
                    mask_tensor = torch.tensor(mask_data).bool().to(device)

                    post_processed_tensor = torch.from_numpy(pred_thresh).float().to(device)

                    metrics = Evaluator().estimate_metrics(post_processed_tensor, mask_tensor, threshold=0.5)

                    # 转换 metrics 为纯数值
                    metrics_val = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
                    local_metrics[image_path.name] = metrics_val

                    # === 恢复打印逻辑 ===
                    # 使用 tqdm.write 可以避免打断进度条显示
                    msg = f"[GPU {gpu_id}] {image_path.name} | Dice: {metrics_val['dice']:.4f} | clDice: {metrics_val['cldice']:.4f}"
                    iterator.write(msg)

    # 存入结果
    return_dict[rank] = local_metrics


@hydra.main(config_path="../configs", config_name="tem_infer", version_base="1.3.2")
def main(cfg):
    # 1. 获取所有数据路径
    all_image_paths, all_mask_paths = get_paths_nested(cfg)
    total_samples = len(all_image_paths)

    # 2. 获取可用 GPU 列表
    if not cfg.get("gpus"):
        logger.warning("未配置 gpus 列表，将尝试使用 cuda:0")
        target_gpus = [0]
    else:
        target_gpus = list(cfg.gpus)

    num_gpus = len(target_gpus)
    logger.info(f"即将使用 {num_gpus} 个 GPU: {target_gpus}")

    # 3. 数据分片
    # 将文件列表切分成 num_gpus 份
    chunk_size = math.ceil(total_samples / num_gpus)
    chunks_img = [all_image_paths[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    if all_mask_paths:
        chunks_mask = [all_mask_paths[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
    else:
        chunks_mask = [None] * len(chunks_img)

    # 4. 启动多进程
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    logger.info(f"开始多 GPU 推理 (Total Samples: {total_samples})...")

    for rank, gpu_id in enumerate(target_gpus):
        if rank >= len(chunks_img):
            break # 防止 GPU 数多于数据块数

        p = mp.Process(
            target=run_inference_worker,
            args=(
                rank,
                gpu_id,
                chunks_img[rank],
                chunks_mask[rank],
                cfg,
                return_dict
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 5. 汇总结果
    logger.info("所有进程已完成，正在汇总指标...")

    final_metrics_dict = {}
    for rank, metrics in return_dict.items():
        final_metrics_dict.update(metrics)

    if final_metrics_dict:
        mean_metrics = calculate_mean_metrics(list(final_metrics_dict.values()), round_to=cfg.round_to)

        logger.info("=" * 60)
        logger.info(f"FINAL GLOBAL AVERAGE METRICS ({len(final_metrics_dict)} cases):")
        logger.info("=" * 60)

        for key in sorted(mean_metrics.keys()):
            val = mean_metrics[key]
            val = val.item() if hasattr(val, 'item') else val
            logger.info(f"Mean {key:<25}: {val:.4f}")
        logger.info("=" * 60)
    else:
        logger.info("没有产生评估指标 (可能未提供标签或 mask_paths 为空)")

    logger.info("Global Inference finished.")

if __name__ == "__main__":
    main()