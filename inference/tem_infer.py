"""
多GPU模型推理脚本 (支持 clDice 打印与记录)
功能:
1. 自动切分数据到多卡并行推理。
2. 汇总结果并在控制台打印 (新增 clDice 显示)。
3. 生成标准化的 CSV 实验报告，包含配置元数据、单例得分及最终平均分。
4. 报告统一保存在脚本运行目录下的 local_results/tem_infer/{dataset_name} 文件夹中。
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

warnings.filterwarnings("ignore")
# 配置多进程启动方式，CUDA必须使用 spawn
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)


def save_csv_report(final_metrics_dict, mean_metrics, cfg, dataset_name):
    """
    【核心修改】生成标准化的实验报告 CSV
    保存位置: {原始运行目录}/local_results/tem_infer/{dataset_name}/
    命名格式: tem_infer_{数据集}_{时间}.csv
    内容结构: 配置信息 -> 详细数据 -> 平均指标
    """
    # 1. 确定保存路径
    try:
        project_root = Path(hydra.utils.get_original_cwd())
    except:
        project_root = Path.cwd()

    csv_dir = project_root / "local_results" / "tem_infer" / dataset_name
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 2. 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    d_name = dataset_name if dataset_name else "UnknownData"
    filename = f"tem_infer_{d_name}_{timestamp}_{cfg.shot_name}shot.csv"
    save_path = csv_dir / filename

    # 3. 准备数据
    sorted_keys = sorted(final_metrics_dict.keys())
    metric_names = []
    if sorted_keys:
        # 获取第一个样本的所有指标名称 (例如: ['dice', 'cldice', 'iou'])
        metric_names = list(final_metrics_dict[sorted_keys[0]].keys())
        # 尝试将 dice 和 cldice 排在前面，方便查看
        priority_keys = ['dice', 'cldice', 'clDice', 'iou']
        metric_names.sort(key=lambda x: (priority_keys.index(x) if x in priority_keys else 999, x))

    try:
        with open(save_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # --- Part 1: 实验配置元数据 (Metadata) ---
            writer.writerow(["### Experiment Configuration Record ###"])
            writer.writerow(["Script Name", "tem_infer.py"])
            writer.writerow(["Date", timestamp])
            writer.writerow(["Dataset Name", d_name])
            writer.writerow(["Model Checkpoint", cfg.ckpt_path])
            writer.writerow(["Input Data Path", cfg.image_path])
            writer.writerow(["Output Mask Path", cfg.output_folder])
            writer.writerow(["TTA Scales", str(cfg.tta.scales)])
            writer.writerow(["TTA Invert", f"{cfg.tta.invert} (Thresh: {cfg.tta.invert_mean_thresh})"])
            writer.writerow(["Patch / Batch", f"{cfg.patch_size} / {cfg.batch_size}"])
            writer.writerow(["Merging Strategy", f"Max: {cfg.merging.max}, Thresh: {cfg.merging.threshold}"])

            post_str = f"SmallObj:{cfg.post.small_objects_min_size}" if cfg.post.apply else "None"
            if cfg.post.get('keep_largest_vessels'): post_str += f", KeepLargest:{cfg.post.num_largest_vessels}"
            writer.writerow(["Post Processing", post_str])

            writer.writerow([]) # 空行分隔

            # --- Part 2: 详细得分 (Detailed Scores) ---
            writer.writerow(["### Detailed Metrics per Case ###"])
            if sorted_keys:
                # 表头
                headers = ["Case Name"] + metric_names
                writer.writerow(headers)

                # 数据行
                for name in sorted_keys:
                    row = [name] + [final_metrics_dict[name].get(k, "") for k in metric_names]
                    writer.writerow(row)
            else:
                writer.writerow(["No metrics calculated (Missing masks?)"])

            writer.writerow([]) # 空行分隔

            # --- Part 3: 平均指标大集合 (Aggregated Metrics) ---
            writer.writerow(["### Final Aggregated Metrics ###"])
            if mean_metrics:
                # 写入两行：一行是指标名，一行是平均值
                # 按照 metric_names 的顺序写入平均值
                sorted_mean_keys = [k for k in metric_names if k in mean_metrics]
                # 加上原本在 mean_metrics 但不在 metric_names 里的其他键
                for k in mean_metrics.keys():
                    if k not in sorted_mean_keys:
                        sorted_mean_keys.append(k)

                writer.writerow(["Metric"] + sorted_mean_keys)
                writer.writerow(["Average"] + [mean_metrics.get(k, 0) for k in sorted_mean_keys])

        logger.info(f"✅ 实验报告已保存至: {save_path}")
        logger.info(f"   (包含了实验配置、{len(sorted_keys)}个样本的详细得分以及最终平均值)")

    except Exception as e:
        logger.error(f"❌ 保存 CSV 报告失败: {e}")


def load_model(cfg, device):
    """加载模型权重"""
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    try:
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

            # --- 计算指标 ---
            if mask_paths:
                if mask_paths[idx] is not None:
                    mask_data = image_reader_writer.read_images(mask_paths[idx])[0]
                    mask_tensor = torch.tensor(mask_data).bool().to(device)

                    post_processed_tensor = torch.from_numpy(pred_thresh).float().to(device)

                    metrics = Evaluator().estimate_metrics(post_processed_tensor, mask_tensor, threshold=0.5)

                    metrics_val = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
                    local_metrics[image_path.name] = metrics_val

                    # 【修改点】新增打印 clDice
                    # 优先获取 cldice 或 clDice，如果都没有则为 0
                    dice_val = metrics_val.get('dice', 0)
                    cldice_val = metrics_val.get('cldice', metrics_val.get('clDice', 0))

                    msg = f"[GPU {gpu_id}] {image_path.name} | Dice: {dice_val:.4f} | clDice: {cldice_val:.4f}"
                    iterator.write(msg)

    # 存入结果
    return_dict[rank] = local_metrics


import re
from pathlib import Path


def auto_infer_paths(cfg):
    """
    自动路径推导逻辑：
    1. 如果 image_path 或 output_folder 是 "auto" 或 None，则根据 ckpt_path 推导。
    2. 如果是具体路径，则保留原值，不修改。
    """
    # 获取当前配置的值 (转为字符串并小写，防止写成 "Auto")
    raw_img_path = str(cfg.get("image_path", "auto")).strip()
    raw_out_path = str(cfg.get("output_folder", "auto")).strip()

    # 判断是否需要自动推导
    need_infer_img = raw_img_path.lower() in ["auto", "none", "null"]
    need_infer_out = raw_out_path.lower() in ["auto", "none", "null"]

    # 如果两个都指定了具体路径，直接返回，不浪费时间解析
    if not need_infer_img and not need_infer_out:
        return cfg

    # --- 开始解析 ckpt_path ---
    ckpt_path = Path(cfg.ckpt_path)
    parts = ckpt_path.parts

    try:
        # 1. 定位 "checkpoints" 文件夹的位置
        ckpt_idx = parts.index("checkpoints")
    except ValueError:
        logger.warning("⚠️ 无法自动推导路径：权重路径中未包含 'checkpoints' 文件夹。将维持原始配置。")
        return cfg

    # 2. 推导 Project Root (local_results 的上一级)
    # parts[:ckpt_idx] 是 .../local_results/
    # parts[:ckpt_idx-1] 是 .../Project/Base-model/
    project_root = Path(*parts[:ckpt_idx - 1])

    # 3. 提取数据集名称 (checkpoints 和 run_folder 之间的部分)
    # 结构: .../checkpoints/{数据集}/{子数据集}/{运行文件夹}/{权重文件}
    # parts[-2] 是运行文件夹 (例如 base_loss_3shot_...)
    run_folder_name = parts[-2]
    dataset_rel_parts = parts[ckpt_idx + 1: -2]
    dataset_rel_path = Path(*dataset_rel_parts)

    # 4. 提取 Shot 数 (用于输出文件夹命名)
    match = re.search(r'(\d+)shot', run_folder_name)
    shot_num = match.group(1) if match else "0"

    # --- 执行赋值 ---

    # (A) 自动推导 image_path
    if need_infer_img:
        # 规则: 项目根目录/datasets/{数据集路径}/test
        autogen_image_path = project_root / "datasets" / dataset_rel_path / "test"
        cfg.image_path = str(autogen_image_path)
        logger.info(f"⚡ [Auto] Image Path 推导为: {cfg.image_path}")
    else:
        logger.info(f"📍 [Manual] Image Path 使用指定路径: {cfg.image_path}")

    # (B) 自动推导 output_folder
    if need_infer_out:
        # 规则: 项目根目录/local_results/output/{数据集路径}/{shot}_shot_test
        autogen_output_folder = project_root / "local_results" / "output" / dataset_rel_path / f"{shot_num}_shot_test"
        cfg.output_folder = str(autogen_output_folder)
        logger.info(f"⚡ [Auto] Output Folder 推导为: {cfg.output_folder}")
    else:
        logger.info(f"📍 [Manual] Output Folder 使用指定路径: {cfg.output_folder}")

    cfg.shot_name= shot_num

    return cfg



@hydra.main(config_path="../configs", config_name="tem_infer", version_base="1.3.2")
def main(cfg):

    # 【第一步】调用自动推导函数
    cfg = auto_infer_paths(cfg)

    # 1. 获取所有数据路径
    all_image_paths, all_mask_paths = get_paths_nested(cfg)
    total_samples = len(all_image_paths)

    # 提取数据集名称（用于CSV命名）
    try:
        dataset_name = Path(cfg.image_path).parent.name
    except:
        dataset_name = "Dataset"

    logger.info("=" * 80)
    logger.info(f"🚀 开始推理任务 (Dataset: {dataset_name})")
    logger.info(f"📂 权重路径: {cfg.ckpt_path}")
    logger.info(f"📂 保存CSV至: ./local_results/tem_infer/{dataset_name}")
    logger.info("=" * 80)

    # 2. 获取可用 GPU 列表
    if not cfg.get("gpus"):
        logger.warning("未配置 gpus 列表，将尝试使用 cuda:0")
        target_gpus = [0]
    else:
        target_gpus = list(cfg.gpus)

    num_gpus = len(target_gpus)

    # 3. 数据分片
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

    for rank, gpu_id in enumerate(target_gpus):
        if rank >= len(chunks_img):
            break

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

    # 5. 汇总结果与记录
    logger.info("所有进程已完成，正在汇总指标...")

    final_metrics_dict = {}
    for rank, metrics in return_dict.items():
        final_metrics_dict.update(metrics)

    if final_metrics_dict:
        # 计算总体平均
        mean_metrics = calculate_mean_metrics(list(final_metrics_dict.values()), round_to=cfg.round_to)

        # 打印控制台简报
        logger.info("=" * 60)
        logger.info(f"🏁 FINAL GLOBAL AVERAGE METRICS ({len(final_metrics_dict)} cases):")
        logger.info("=" * 60)
        # 打印所有平均指标
        for key in sorted(mean_metrics.keys()):
            val = mean_metrics[key]
            val = val.item() if hasattr(val, 'item') else val
            logger.info(f"Mean {key:<25}: {val:.4f}")
        logger.info("=" * 60)

        # 保存增强版 CSV 报告 (自动包含 cldice)
        save_csv_report(final_metrics_dict, mean_metrics, cfg, dataset_name)

    else:
        logger.info("没有产生评估指标 (可能未提供标签或 mask_paths 为空)")

    logger.info("Global Inference finished.")

if __name__ == "__main__":
    main()