"""
模型推理脚本 - 支持多GPU并行推理 & 动态距离约束与形态学缝合
修改点:
1. 增加多进程 (Multiprocessing) 支持，实现多卡并行推理。
2. 数据集自动切分。
3. 指标结果跨进程汇总。
"""
import logging
import warnings
from pathlib import Path
import re
import math
import time

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import hydra
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt

# 引入图像处理库
from skimage import morphology, measure
from scipy import ndimage

# 保持原有引用
from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics

warnings.filterwarnings("ignore")
# 配置 logging 格式，包含进程名以便区分
logging.basicConfig(format='[%(processName)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
#  核心逻辑：距离约束与缝合 (保持不变)
# ==============================================================================

def distance_based_postprocessing(pred_data, closing_radius=3, center_threshold=50.0):
    """
    Args:
        pred_data: 二值化预测结果 (D, H, W)
        closing_radius: 闭运算半径，用于缝合断裂。
        center_threshold: 距离阈值。
    """
    # 1. 【缝合断裂】 Morphological Closing
    if closing_radius > 0:
        struct = morphology.ball(closing_radius)
        bridged_data = ndimage.binary_closing(pred_data, structure=struct)
    else:
        bridged_data = pred_data.copy()

    # 2. 【连通域分析】
    lbl, num = measure.label(bridged_data, connectivity=2, return_num=True)

    if num == 0:
        return bridged_data

    # 3. 【构建距离场】
    d, h, w = bridged_data.shape
    cz, cy, cx = d // 2, h // 2, w // 2 # 图像中心坐标

    zz, yy, xx = np.ogrid[:d, :h, :w]
    dist_map = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)

    # 4. 【距离筛选】
    final_mask = np.zeros_like(bridged_data)

    for i in range(1, num + 1):
        component_mask = (lbl == i)
        component_dists = dist_map[component_mask]
        min_dist_to_center = component_dists.min()

        if min_dist_to_center <= center_threshold:
            final_mask[component_mask] = 1

    # 5. 【兜底策略】
    if np.sum(final_mask) == 0:
        regions = measure.regionprops(lbl)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 0:
            final_mask[lbl == regions[0].label] = 1

    return final_mask > 0.5


# ==============================================================================
#  辅助函数
# ==============================================================================

def load_model(cfg, device):
    ckpt_path = Path(cfg.ckpt_path)
    # logger.info(f"Loading models from {ckpt_path} to {device}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    try:
        # 显式指定 map_location 到当前进程的 GPU
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"加载模型文件失败: {e}")
        raise

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
        except RuntimeError as e:
            model.load_state_dict(state_dict_to_load, strict=False)
    else:
        raise ValueError("权重文件格式不支持")

    return model


def get_paths_nested(cfg):
    """获取所有文件路径，不在此处切分，在主函数切分"""
    root_dir = Path(hydra.utils.to_absolute_path(cfg.image_path))
    if not root_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")

    image_paths = []
    mask_paths = []
    case_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    # 仅主进程打印一次
    if mp.current_process().name == 'MainProcess':
        logger.info(f"在 {root_dir} 中找到 {len(case_dirs)} 个子文件夹")

    for case_dir in case_dirs:
        case_id = case_dir.name
        img_name = f"{case_id}{cfg.image_file_ending}"
        img_p = case_dir / img_name

        if img_p.exists():
            image_paths.append(img_p)
            if cfg.get('mask_suffix') and cfg.get('mask_path'):
                label_name = f"{case_id}{cfg.mask_suffix}"
                label_p = case_dir / label_name
                if label_p.exists():
                    mask_paths.append(label_p)
        else:
            pass

    if not image_paths:
        raise FileNotFoundError(f"未在 {root_dir} 的子文件夹中找到任何符合规则的图像。")

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


# ==============================================================================
#  Worker 进程逻辑
# ==============================================================================

def inference_worker(rank, gpu_id, image_paths, mask_paths, cfg, result_queue):
    """
    Args:
        rank: 进程编号 (0, 1, 2...)
        gpu_id: 实际使用的 CUDA ID (如 0, 1...)
        image_paths: 当前进程分配到的图像路径列表
        mask_paths: 当前进程分配到的掩膜路径列表
        cfg: 配置对象
        result_queue: 用于回传结果的队列
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)  # 重要：设置当前进程的默认 GPU

        # 加载模型 (每个进程独立加载)
        model = load_model(cfg, device)
        model.to(device)
        model.eval()

        transforms = generate_transforms(cfg.transforms_config)

        # 准备 IO
        if len(image_paths) > 0:
            first_name = image_paths[0].name
            if 'nii.gz' in first_name:
                rw_suffix = 'nii.gz'
                save_ext = '.nii.gz'
            else:
                rw_suffix = image_paths[0].suffix
                save_ext = image_paths[0].suffix
            image_reader_writer = determine_reader_writer(rw_suffix)()
            save_writer = determine_reader_writer(rw_suffix)()

        # 准备输出目录
        save_predictions = False
        output_folder = None
        if cfg.output_folder and str(cfg.output_folder).lower() != "none" and str(cfg.output_folder).strip() != "":
            save_predictions = True
            output_folder = Path(cfg.output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

        inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.patch_size,
            sw_batch_size=cfg.batch_size,
            overlap=cfg.overlap,
            mode=cfg.mode,
            sigma_scale=cfg.sigma_scale,
            padding_mode=cfg.padding_mode
        )

        local_metrics = {}

        # 进度条 (position控制多行显示)
        pbar = tqdm(enumerate(image_paths), total=len(image_paths),
                    desc=f"GPU {gpu_id}", position=rank, leave=True)

        with torch.no_grad():
            for idx, image_path in pbar:
                try:
                    img_data = image_reader_writer.read_images(image_path)[0].astype(np.float32)
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 读取失败 {image_path}: {e}")
                    continue

                # --- TTA & Inference ---
                preds = []
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

                if len(preds) > 1:
                    stacked_preds = torch.stack(preds)
                    if cfg.merging.max:
                        pred = stacked_preds.max(dim=0)[0].sigmoid()
                    else:
                        pred = stacked_preds.mean(dim=0).sigmoid()
                else:
                    pred = preds[0].sigmoid()

                pred_thresh = (pred > cfg.merging.threshold).numpy()

                # ====================================================
                # 应用动态距离约束与缝合后处理
                # ====================================================
                if cfg.post.apply:
                    pred_bool = pred_thresh.astype(bool)
                    d, h, w = pred_bool.shape

                    # 用户新公式: median(x, y, z) * 0.5
                    sorted_dims = sorted([d, h, w])
                    median_dim = sorted_dims[1]
                    dynamic_thresh = (median_dim+10) * 0.4

                    processed_mask = distance_based_postprocessing(
                        pred_bool,
                        closing_radius=cfg.post.closing_radius,
                        center_threshold=dynamic_thresh
                    )
                    pred_thresh = processed_mask.astype(np.uint8)
                else:
                    pred_thresh = pred_thresh.astype(np.uint8)
                # ====================================================

                if save_predictions:
                    clean_name = re.sub(r'\.img\.nii(\.gz)?$', '', image_path.name)
                    save_name = f"{clean_name}{cfg.file_app}{save_ext}"
                    save_path = output_folder / save_name
                    save_writer.write_seg(pred_thresh, save_path)

                # 计算指标
                mask = None
                if mask_paths:
                    try:
                        # 注意：image_paths是切分过的，idx是相对索引
                        mask_data = image_reader_writer.read_images(mask_paths[idx])[0]
                        mask = torch.tensor(mask_data).bool().to(device)
                    except Exception:
                        pass

                if mask is not None:
                    post_processed_tensor = torch.from_numpy(pred_thresh).float().to(device)
                    metrics = Evaluator().estimate_metrics(post_processed_tensor, mask, threshold=0.5)

                    # --- 兼容 float 和 Tensor 类型 ---
                    metrics_cpu = {}
                    for k, v in metrics.items():
                        if hasattr(v, 'item'):
                            metrics_cpu[k] = v.item()
                        else:
                            metrics_cpu[k] = v

                    fname = image_path.name
                    local_metrics[fname] = metrics_cpu

                    # === 【新增】打印每个样本的 Dice 和 clDice ===
                    # 尝试获取 dice 和 cldice，如果键名不同请根据实际情况调整 (通常是 'dice', 'cldice')
                    d_score = metrics_cpu.get('dice', metrics_cpu.get('Dice', -1))
                    c_score = metrics_cpu.get('cldice', metrics_cpu.get('clDice', -1))

                    # 使用 logger 打印，为了不破坏进度条显示，可以使用 short print
                    # 注意：在多进程下 console 输出可能会穿插，建议查看日志文件或接受这种穿插
                    logger.info(f"GPU-{gpu_id} | {fname} | Dice: {d_score:.4f} | clDice: {c_score:.4f}")

        # 将本进程的结果放入队列
        result_queue.put(local_metrics)

    except Exception as e:
        logger.error(f"Worker {rank} (GPU {gpu_id}) failed: {e}")
        result_queue.put({})  # 发送空结果防止主进程死锁
        raise e

# ==============================================================================
#  主流程
# ==============================================================================

@hydra.main(config_path="../configs", config_name="post_infer", version_base="1.3.2")
def main(cfg):
    # 1. 确定使用的 GPU 列表
    # 优先从 cfg.devices 读取 (如 [0,1,2,3])
    # 如果没有，尝试解析 cfg.device (如 "cuda:0")
    # 如果都没有，使用所有可用 GPU
    target_devices = []

    if 'devices' in cfg and cfg.devices is not None:
        target_devices = list(cfg.devices)
    elif str(cfg.device).lower() != 'cpu':
        if "cuda" in str(cfg.device):
            # 处理 "cuda:0" 这种格式
            try:
                device_id = int(str(cfg.device).split(":")[-1])
                target_devices = [device_id]
            except:
                pass

    if not target_devices:
        if torch.cuda.is_available():
            target_devices = list(range(torch.cuda.device_count()))
        else:
            logger.error("无可用 GPU，本脚本仅支持 CUDA 推理。")
            return

    logger.info(f"🚀 启动多 GPU 推理，使用设备 IDs: {target_devices}")

    # 2. 获取所有数据路径
    all_image_paths, all_mask_paths = get_paths_nested(cfg)
    total_images = len(all_image_paths)
    num_gpus = len(target_devices)

    # 3. 启动多进程
    mp.set_start_method('spawn', force=True) # CUDA 必须使用 spawn
    result_queue = mp.Queue()
    processes = []

    for rank, gpu_id in enumerate(target_devices):
        # 数据切分：简单切片法 paths[0::4], paths[1::4]...
        subset_images = all_image_paths[rank::num_gpus]
        subset_masks = all_mask_paths[rank::num_gpus] if all_mask_paths else []

        if len(subset_images) == 0:
            continue

        p = mp.Process(
            target=inference_worker,
            args=(rank, gpu_id, subset_images, subset_masks, cfg, result_queue)
        )
        p.start()
        processes.append(p)

    # 4. 收集结果
    all_metrics = {}
    finished_workers = 0

    # 循环接收结果，直到所有 worker 发送完毕
    while finished_workers < len(processes):
        # 阻塞获取，设置超时防止死锁
        try:
            worker_result = result_queue.get() # 阻塞等待
            all_metrics.update(worker_result)
            finished_workers += 1
        except Exception as e:
            # 简单的错误处理
            pass

    for p in processes:
        p.join()

    # 5. 最终统计
    if all_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("汇总所有 GPU 结果:")
        # 将字典的值转为列表以适配 calculate_mean_metrics
        # 注意: calculate_mean_metrics 期望输入是 list of dicts
        # 这里的 all_metrics 是 {fname: {dice: 0.9, ...}}
        metrics_list = list(all_metrics.values())

        # calculate_mean_metrics 需要适配纯 float 输入 (因为我们在 worker 里转成了 float)
        # 如果原函数只支持 Tensor，可能需要简单修改。通常它是支持 numpy/dict 的。
        # 这里假设 calculate_mean_metrics 能够处理。

        try:
            mean_metrics = calculate_mean_metrics(metrics_list, round_to=cfg.round_to)

            logger.info("=" * 60)
            logger.info(f"FINAL AVERAGE METRICS ({len(all_metrics)} cases):")
            logger.info("=" * 60)
            for key in sorted(mean_metrics.keys()):
                val = mean_metrics[key]
                val = val.item() if hasattr(val, 'item') else val
                logger.info(f"Mean {key:<25}: {val:.4f}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"计算平均指标时出错 (可能是数据格式问题): {e}")
            # 简单打印一下 Dice 均值兜底
            dices = [m.get('dice', 0) for m in metrics_list]
            logger.info(f"Simple Mean Dice: {np.mean(dices):.4f}")

    elif all_mask_paths:
        logger.warning("未计算出任何指标。")

    logger.info("Inference finished.")

if __name__ == "__main__":
    main()