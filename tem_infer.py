"""
模型推理脚本 - 适配嵌套文件夹结构
结构: Root/CaseID/CaseID.img.nii.gz & CaseID.label.nii.gz
新增功能: output_folder 为空时不保存文件，仅评估
修改功能: 最后打印所有平均评估指标
"""
import logging
import warnings
from pathlib import Path
import re

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist
from skimage.measure import label, regionprops

# 保持原有的引用不变
from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_model(cfg, device):
    """
    加载模型权重 (保持不变)
    """
    ckpt_path = Path(cfg.ckpt_path)
    logger.info(f"Loading models from {ckpt_path}.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"加载模型文件失败: {e}")
        raise

    model = hydra.utils.instantiate(cfg.model)
    state_dict_to_load = None

    if 'state_dict' in ckpt:
        logger.info("检测到PyTorch Lightning格式的.ckpt文件")
        state_dict = ckpt['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除常见前缀
            key = key.replace('models.', '').replace('net.', '').replace('module.', '')
            new_state_dict[key] = value
        state_dict_to_load = new_state_dict

    elif isinstance(ckpt, dict) and any(key.startswith(('encoder', 'decoder', 'backbone')) for key in ckpt.keys()):
        state_dict_to_load = ckpt
    else:
        state_dict_to_load = ckpt

    if state_dict_to_load is not None:
        try:
            model.load_state_dict(state_dict_to_load, strict=True)
            logger.info("模型权重严格加载成功")
        except RuntimeError as e:
            logger.warning(f"严格加载失败: {e}，尝试非严格加载")
            model.load_state_dict(state_dict_to_load, strict=False)
    else:
        raise ValueError("权重文件格式不支持")

    return model


def get_paths_nested(cfg):
    """
    专门适配嵌套结构的路径获取函数 (保持不变)
    """
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
        else:
            logger.debug(f"Case {case_id}: 未找到图像文件 {img_name}，跳过。")

    if not image_paths:
        raise FileNotFoundError(f"未在 {root_dir} 的子文件夹中找到任何符合规则的图像。")

    if mask_paths and len(mask_paths) != len(image_paths):
        logger.warning(f"警告：图像数量 ({len(image_paths)}) 与 标签数量 ({len(mask_paths)}) 不一致。")
        if cfg.get('strict_matching', True):
            logger.error("严格模式开启：标签不匹配，停止运行。")
            raise ValueError("标签数量不匹配")
        else:
            logger.warning("非严格模式：将禁用评估功能。")
            mask_paths = None

    if not mask_paths:
        return image_paths, None

    return image_paths, mask_paths


def resample(image, factor=None, target_shape=None):
    """保持不变"""
    if factor == 1: return image
    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


def keep_largest_vessels(prediction, num_vessels=3):
    """保持不变"""
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
    """保持不变"""
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


@hydra.main(config_path="configs", config_name="tem_infer", version_base="1.3.2")
def main(cfg):
    """
    推理主函数
    """
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger.info(f"Using device {cfg.device}.")
    if(cfg.device == "cuda"):
        cfg.device = "cuda:0"
    device = torch.device(cfg.device)

    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    transforms = generate_transforms(cfg.transforms_config)

    image_paths, mask_paths = get_paths_nested(cfg)

    # Output Folder Logic
    save_predictions = False
    output_folder = None
    if cfg.output_folder and str(cfg.output_folder).lower() != "none" and str(cfg.output_folder).strip() != "":
        save_predictions = True
        output_folder = Path(cfg.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"预测结果将保存至: {output_folder}")
    else:
        logger.info("未设置 output_folder 或设置为空。将跳过保存预测文件，仅进行评估。")

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

    metrics_dict = {}

    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Inference"):
            preds = []
            try:
                img_data = image_reader_writer.read_images(image_path)[0].astype(np.float32)
            except Exception as e:
                logger.error(f"读取图像失败 {image_path}: {e}")
                continue

            for scale in cfg.tta.scales:
                image_tensor = transforms(img_data)
                if isinstance(image_tensor, np.ndarray):
                    image_tensor = torch.from_numpy(image_tensor)
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                image = image_tensor.unsqueeze(0).to(device)

                mask = None
                if mask_paths and idx < len(mask_paths):
                    mask_data = image_reader_writer.read_images(mask_paths[idx])[0]
                    mask = torch.tensor(mask_data).bool()

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

            if cfg.post.apply:
                pred_thresh = remove_small_objects(
                    pred_thresh.astype(bool),
                    min_size=cfg.post.small_objects_min_size,
                    connectivity=cfg.post.small_objects_connectivity
                )
            if cfg.post.get('keep_largest_vessels', False):
                pred_thresh = keep_largest_vessels(
                    pred_thresh.astype(int),
                    num_vessels=cfg.post.num_largest_vessels
                )
            if cfg.post.get('keep_closest_vessels', False):
                pred_thresh = keep_closest_vessels(
                    pred_thresh.astype(int),
                    num_vessels=cfg.post.num_closest_vessels
                )

            if save_predictions:
                clean_name = image_path.name.replace('.img.nii.gz', '').replace('.nii.gz', '')
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
                save_path = output_folder / save_name
                save_writer.write_seg(pred_thresh.astype(np.uint8), save_path)

            if mask is not None:
                post_processed_tensor = torch.from_numpy(pred_thresh).float().to(device)
                metrics = Evaluator().estimate_metrics(post_processed_tensor, mask, threshold=0.5)

                fname = image_path.name
                # 单个样本日志也可以选择打印全部，这里为了简洁只保留了关键的两个
                # 如果想看全部，可以将这里也改成遍历打印
                logger.info("=" * 40)
                logger.info(f"Dice of {fname}: {metrics['dice'].item():.4f}")
                logger.info(f"clDice of {fname}: {metrics['cldice'].item():.4f}")
                metrics_dict[fname] = metrics

    # --- 最终平均指标汇总 ---
    if metrics_dict:
        mean_metrics = calculate_mean_metrics(list(metrics_dict.values()), round_to=cfg.round_to)

        logger.info("=" * 60)
        logger.info(f"FINAL AVERAGE METRICS ({len(metrics_dict)} cases):")
        logger.info("=" * 60)

        # 遍历所有指标并打印
        # 对键进行排序，使输出整齐
        for key in sorted(mean_metrics.keys()):
            val = mean_metrics[key]
            # 兼容 Tensor 和 普通数值
            val = val.item() if hasattr(val, 'item') else val
            logger.info(f"Mean {key:<25}: {val:.4f}")

        logger.info("=" * 60)

    elif mask_paths:
        logger.warning("未计算出任何指标，请检查数据或标签是否匹配。")

    logger.info("Inference finished.")


if __name__ == "__main__":
    main()