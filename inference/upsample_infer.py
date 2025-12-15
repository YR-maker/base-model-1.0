"""
upsample_infer.py
基于 tem_infer.py 修改。
保留所有原有配置和流程，仅增加：
1. 推理前强制上采样 (inference_target_shape)。
2. 推理后下采样回原分辨率。
3. 保存两份结果 (原尺寸 + 上采样尺寸)。
"""
import logging
import warnings
from pathlib import Path
import math
import os
import csv
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import hydra
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from omegaconf import OmegaConf

# 保持原有的引用不变
from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics
# 直接复用 tem_infer 中的函数，确保逻辑一致
from tem_infer import (
    save_csv_report,
    load_model,
    get_paths_nested,
    resample, # TTA用的resample
    keep_largest_vessels,
    keep_closest_vessels,
    auto_infer_paths
)

warnings.filterwarnings("ignore")
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)


def resize_tensor_force(tensor, target_shape, mode="trilinear"):
    """
    强制缩放到指定尺寸 [D, H, W]
    修复: 强制将 OmegaConf 的 ListConfig 转换为标准的 python list[int]
    """
    # 无论传入的是 ListConfig, tuple 还是 list，先强制转为 list，再转 int
    if target_shape is not None:
        target_shape = [int(x) for x in list(target_shape)]

    input_ndim = tensor.ndim
    # 增加维度以适配 interpolate (需要 B, C, D, H, W)
    if input_ndim == 3:  # D, H, W -> 1, 1, D, H, W
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif input_ndim == 4:  # C, D, H, W -> 1, C, D, H, W
        tensor = tensor.unsqueeze(0)

    # trilinear 插值通常 align_corners=False
    align = False if mode != 'nearest' else None

    # 执行插值
    resized = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=align)

    # 还原维度
    if input_ndim == 3:
        return resized.squeeze(0).squeeze(0)
    elif input_ndim == 4:
        return resized.squeeze(0)
    return resized

def run_upsample_inference_worker(rank, gpu_id, image_paths, mask_paths, cfg, return_dict):
    """
    修改后的 Worker：支持上采样 -> 推理 -> 保存中间结果 -> 下采样 -> 评估
    """
    device = torch.device(f"cuda:{gpu_id}")

    # 设置随机种子 (保持 tem_infer 逻辑)
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

    # 确定 I/O (保持 tem_infer 逻辑)
    if not image_paths: return
    first_name = image_paths[0].name
    if 'nii.gz' in first_name:
        rw_suffix = 'nii.gz'
        save_ext = '.nii.gz'
    else:
        rw_suffix = image_paths[0].suffix
        save_ext = image_paths[0].suffix

    image_reader_writer = determine_reader_writer(rw_suffix)()
    save_writer = determine_reader_writer(rw_suffix)()

    # 滑动窗口推断器 (保持 tem_infer 逻辑)
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

    desc = f"GPU {gpu_id}"
    iterator = tqdm(enumerate(image_paths), total=len(image_paths), desc=desc, position=rank, leave=True)

    # 获取目标尺寸配置
    target_shape_cfg = cfg.get("inference_target_shape", None)

    with torch.no_grad():
        for idx, image_path in iterator:
            preds = []
            try:
                img_data = image_reader_writer.read_images(image_path)[0].astype(np.float32)
            except Exception as e:
                print(f"[GPU {gpu_id}] 读取失败 {image_path}: {e}")
                continue

            # 1. 基础预处理
            image_tensor = transforms(img_data) # (C, D, H, W)
            if isinstance(image_tensor, np.ndarray):
                image_tensor = torch.from_numpy(image_tensor)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # --- 【修改点 1】记录原始尺寸并强制上采样 ---
            original_spatial_shape = image_tensor.shape[1:] # D, H, W

            # 如果配置了 target_shape，强制 Resize 输入图像
            if target_shape_cfg:
                image_tensor = resize_tensor_force(image_tensor, target_shape_cfg, mode="trilinear")

            # 移至 GPU
            image_base = image_tensor.unsqueeze(0).to(device) # (1, C, D', H', W')

            # --- TTA & 推理 (保持 tem_infer 逻辑，但在上采样后的图上进行) ---
            for scale in cfg.tta.scales:
                image = image_base.clone()

                if cfg.tta.invert and image.mean() > cfg.tta.invert_mean_thresh:
                    image = 1 - image

                # TTA 的 resample 也是基于当前的 image_base 尺寸
                current_shape = image.shape # (1, C, D', H', W')
                image = resample(image, factor=scale)

                logits = inferer(image, model)
                logits = resample(logits, target_shape=current_shape)
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

            # pred 现在是 Tensor (C, D', H', W')，是上采样后的概率图

            # --- 【修改点 2】保存上采样结果 & 下采样回原尺寸 ---

            # A. 处理上采样结果 (用于保存)
            if save_predictions and target_shape_cfg:
                # 二值化
                pred_upsampled_thresh = (pred > cfg.merging.threshold).numpy().astype(np.uint8)

                # 保存文件名加 _upsampled
                clean_name = image_path.name.replace('.img.nii.gz', '').replace('.nii.gz', '')
                save_name_up = f"{clean_name}_upsampled_{cfg.file_app}pred{save_ext}"
                save_writer.write_seg(pred_upsampled_thresh, output_folder / save_name_up)

            # B. 下采样回原始尺寸 (用于后续标准的保存和评估)
            if target_shape_cfg:
                # 注意：对概率图插值，使用 trilinear
                pred = resize_tensor_force(pred, original_spatial_shape, mode="trilinear")

            # --- 以下完全保持 tem_infer.py 的逻辑 ---

            pred_thresh = (pred > cfg.merging.threshold).numpy()

            # --- 后处理 (保持不变) ---
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

            # --- 保存 (原始尺寸) ---
            if save_predictions:
                clean_name = image_path.name.replace('.img.nii.gz', '').replace('.nii.gz', '')
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
                save_path = output_folder / save_name
                save_writer.write_seg(pred_thresh.astype(np.uint8), save_path)

            # --- 计算指标 (保持不变) ---
            if mask_paths:
                if mask_paths[idx] is not None:
                    # 读取原始 GT (原始尺寸)
                    mask_data = image_reader_writer.read_images(mask_paths[idx])[0]
                    mask_tensor = torch.tensor(mask_data).bool().to(device)

                    # 此时 pred_thresh 已经被 resize 回原始尺寸了，可以直接计算
                    post_processed_tensor = torch.from_numpy(pred_thresh).float().to(device)

                    metrics = Evaluator().estimate_metrics(post_processed_tensor, mask_tensor, threshold=0.5)

                    metrics_val = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
                    local_metrics[image_path.name] = metrics_val

                    dice_val = metrics_val.get('dice', 0)
                    cldice_val = metrics_val.get('cldice', metrics_val.get('clDice', 0))

                    msg = f"[GPU {gpu_id}] {image_path.name} | Dice: {dice_val:.4f} | clDice: {cldice_val:.4f}"
                    iterator.write(msg)

    # 存入结果
    return_dict[rank] = local_metrics


@hydra.main(config_path="../configs", config_name="upsample_infer", version_base="1.3.2")
def main(cfg):
    # 完全复用 tem_infer 的 main 逻辑，除了 worker 函数变了

    cfg = auto_infer_paths(cfg)
    all_image_paths, all_mask_paths = get_paths_nested(cfg)
    total_samples = len(all_image_paths)

    try:
        dataset_name = Path(cfg.image_path).parent.name
    except:
        dataset_name = "Dataset"

    logger.info("=" * 80)
    logger.info(f"🚀 开始 [上采样] 推理任务 (Dataset: {dataset_name})")
    # 打印一下目标尺寸
    logger.info(f"🎯 强制推理尺寸: {cfg.get('inference_target_shape', 'Disabled')}")
    logger.info(f"📂 权重路径: {cfg.ckpt_path}")
    logger.info("=" * 80)

    if not cfg.get("gpus"):
        target_gpus = [0]
    else:
        target_gpus = list(cfg.gpus)

    num_gpus = len(target_gpus)
    chunk_size = math.ceil(total_samples / num_gpus)
    chunks_img = [all_image_paths[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

    if all_mask_paths:
        chunks_mask = [all_mask_paths[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
    else:
        chunks_mask = [None] * len(chunks_img)

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank, gpu_id in enumerate(target_gpus):
        if rank >= len(chunks_img):
            break

        # 使用新的 worker
        p = mp.Process(
            target=run_upsample_inference_worker,
            args=(rank, gpu_id, chunks_img[rank], chunks_mask[rank], cfg, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("所有进程已完成，正在汇总指标...")
    final_metrics_dict = {}
    for rank, metrics in return_dict.items():
        final_metrics_dict.update(metrics)

    if final_metrics_dict:
        mean_metrics = calculate_mean_metrics(list(final_metrics_dict.values()), round_to=cfg.round_to)

        logger.info("=" * 60)
        logger.info(f"🏁 FINAL METRICS ({len(final_metrics_dict)} cases):")
        for key in sorted(mean_metrics.keys()):
            val = mean_metrics[key]
            val = val.item() if hasattr(val, 'item') else val
            logger.info(f"Mean {key:<25}: {val:.4f}")
        logger.info("=" * 60)

        # 复用 tem_infer 的保存报告函数
        save_csv_report(final_metrics_dict, mean_metrics, cfg, dataset_name)
    else:
        logger.info("没有产生评估指标")

if __name__ == "__main__":
    main()