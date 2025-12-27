import logging, warnings, math, os, csv, re, sys

# --- 【核心修复】强力屏蔽警告 (必须放在其他 import 之前) ---
warnings.filterwarnings("ignore")
# 专门针对 monai/pkg_resources 的顽固警告
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
# 确保多进程(mp.spawn)启动的子进程也能屏蔽警告
os.environ["PYTHONWARNINGS"] = "ignore"
# ----------------------------------------------------
from pathlib import Path
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
from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics

warnings.filterwarnings("ignore")
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
logger = logging.getLogger(__name__)


# --- 辅助函数：后处理与图像操作 ---
def resample(img, factor=1, target_shape=None):
    if factor == 1 and target_shape is None: return img
    size = target_shape[-3:] if target_shape else [int(round(s / factor)) for s in img.shape[-3:]]
    return F.interpolate(img, size=size, mode="trilinear", align_corners=False)


def apply_post_processing(pred, cfg):
    """整合后的后处理逻辑"""
    if not cfg.post.apply: return pred
    # 去除小物体
    mask = remove_small_objects(pred.astype(bool), min_size=cfg.post.small_objects_min_size)
    if not (cfg.post.get('keep_largest_vessels') or cfg.post.get('keep_closest_vessels')):
        return mask.astype(int)

    # 连通域筛选 (最大 or 最近)
    lbl, num = label(mask, return_num=True, connectivity=3)
    if num == 0: return mask.astype(int)
    props = regionprops(lbl)

    if cfg.post.get('keep_largest_vessels'):
        targets = sorted(props, key=lambda x: x.area, reverse=True)[:cfg.post.num_largest_vessels]
    elif cfg.post.get('keep_closest_vessels'):
        center = np.array(pred.shape) / 2.0
        targets = sorted(props, key=lambda x: np.linalg.norm(np.array(x.centroid) - center))[
                  :cfg.post.num_closest_vessels]
    else:
        return mask.astype(int)

    out = np.zeros_like(pred)
    for r in targets: out[tuple(r.coords.T)] = 1
    return out


def save_report(metrics, mean, cfg, d_name):
    """精简版CSV报告生成"""
    ts = datetime.now().strftime("%m%d_%H%M")
    path = Path(
        hydra.utils.get_original_cwd()) / "infer_test" / d_name / f"{cfg.shot_name}_{d_name}_{ts}shot.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(metrics.keys())
    headers = list(metrics[keys[0]].keys()) if keys else []

    # 【修改点】定义排序优先级：Dice -> clDice -> NSD -> ASD -> 其他
    priority = ['dice', 'cldice', 'clDice', 'nsd', 'asd']

    # 1. 排序 Detailed 表头
    headers.sort(key=lambda x: (priority.index(x) if x in priority else 99, x))

    # 2. 排序 Summary 键
    mean_keys = sorted(mean.keys(), key=lambda x: (priority.index(x) if x in priority else 99, x))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerows([
            ["### Config ###"], ["Time", ts], ["Model", cfg.ckpt_path],
            ["TTA", f"{cfg.tta.scales} (Inv:{cfg.tta.invert})"], ["Post", f"{cfg.post.apply}"], []
        ])
        if keys:
            w.writerow(["### Details ###"])
            w.writerow(["Case"] + headers)
            w.writerows([[k] + [metrics[k].get(h, "") for h in headers] for k in keys])

        # 使用排序后的 mean_keys 写入 Summary
        w.writerows([[], ["### Summary ###"], ["Metric"] + mean_keys, ["Avg"] + [mean[k] for k in mean_keys]])
    logger.info(f"✅ Report saved: {path}")


# --- 核心逻辑 ---
def load_model(cfg, device):
    model = hydra.utils.instantiate(cfg.model).to(device)
    # 显式添加 weights_only=False 以允许加载包含配置对象的旧版权重文件
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)

    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # 自动去除前缀
    state = {k.replace('model.', '').replace('net.', '').replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.eval()


def worker(rank, gpu, files, masks, cfg, out_dict):
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(cfg.seed + rank)
    try:
        model = load_model(cfg, device)
    except Exception as e:
        return print(f"[GPU {gpu}] Model load fail: {e}")

    inferer = SlidingWindowInfererAdapt(roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap,
                                        mode=cfg.mode)
    trans = generate_transforms(cfg.transforms_config)

    # I/O 后缀逻辑
    fname = files[0].name
    suffix = 'nii.gz' if 'nii.gz' in fname else files[0].suffix
    try:
        io_cls = determine_reader_writer(suffix)
    except ValueError:
        io_cls = determine_reader_writer(suffix.strip('.'))

    rw, output_dir = io_cls(), Path(
        cfg.output_folder) if cfg.output_folder and cfg.output_folder.lower() != "none" else None
    if output_dir: output_dir.mkdir(parents=True, exist_ok=True)

    local_res = {}
    pbar = tqdm(zip(files, masks if masks else [None] * len(files)), total=len(files), desc=f"GPU {gpu}", position=rank)

    with torch.no_grad():
        for img_p, msk_p in pbar:
            try:
                img = rw.read_images(img_p)[0].astype(np.float32)
            except Exception as e:
                print(f"Read Error: {img_p} -> {e}")
                continue

            # TTA 推理
            preds = []
            for s in cfg.tta.scales:
                # 1. 获取增强后的数据
                data_in = trans(img)

                # 2. 如果是 numpy 则转 tensor，如果是 MetaTensor/Tensor 则保持原样
                if isinstance(data_in, np.ndarray):
                    x = torch.from_numpy(data_in)
                else:
                    x = data_in  # 已经是 Tensor/MetaTensor

                # 3. 维度修正：确保形状为 (Batch, Channel, D, H, W)
                if x.ndim == 3:  # (D, H, W) -> 缺 Channel 和 Batch
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.ndim == 4:  # (C, D, H, W) -> 缺 Batch
                    x = x.unsqueeze(0)

                x = x.to(device)

                if cfg.tta.invert and x.mean() > cfg.tta.invert_mean_thresh: x = 1 - x
                orig_sh = x.shape
                logit = resample(inferer(resample(x, s), model), target_shape=orig_sh)
                preds.append(logit.cpu().squeeze().sigmoid())

            # 融合
            if len(preds) > 1:
                pred = torch.stack(preds).max(dim=0)[0] if cfg.merging.max else torch.stack(preds).mean(dim=0)
            else:
                pred = preds[0]

            res = apply_post_processing((pred.numpy() > cfg.merging.threshold).astype(int), cfg)

            # 保存
            if output_dir:
                save_ext = '.nii.gz' if 'nii.gz' in fname else files[0].suffix
                clean_name = img_p.name.replace('.img', '').replace('.nii.gz', '').replace(save_ext, '')
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
                rw.write_seg(res.astype(np.uint8), output_dir / save_name)

            # 评估
            if msk_p:
                m_ts = torch.tensor(rw.read_images(msk_p)[0]).bool().to(device)
                met = Evaluator().estimate_metrics(torch.from_numpy(res).float().to(device), m_ts, threshold=0.5)
                met_v = {k: v.item() if hasattr(v, 'item') else v for k, v in met.items()}
                local_res[img_p.name] = met_v
                pbar.write(
                    f"[GPU {gpu}] {img_p.name} | Dice: {met_v.get('dice', 0):.4f} | clDice: {met_v.get('cldice', met_v.get('clDice', 0)):.4f}")

    out_dict[rank] = local_res


def resolve_paths(cfg):
    """自动路径解析与列表获取"""
    # 1. 自动推导
    if str(cfg.image_path).lower() in ["auto", "none"] or str(cfg.output_folder).lower() in ["auto", "none"]:
        try:
            parts = Path(cfg.ckpt_path).parts
            idx = parts.index("checkpoints")
            proj, ds_rel = Path(*parts[:idx - 1]), Path(*parts[idx + 1:-2])
            shot = re.search(r'(\d+)shot', parts[-2]).group(1) if re.search(r'(\d+)shot', parts[-2]) else "0"

            if str(cfg.image_path).lower() in ["auto", "none"]:
                cfg.image_path = str(proj / "datasets" / ds_rel / "test")
            if str(cfg.output_folder).lower() in ["auto", "none"]:
                cfg.output_folder = str(proj / "local_results/output" / ds_rel / f"{shot}_shot_test")
            cfg.shot_name = shot
            logger.info(f"⚡ Auto Paths: Img={cfg.image_path} | Out={cfg.output_folder}")
        except Exception:
            logger.warning("⚠️ Auto path failed, using original.");
            pass

    # 2. 获取文件列表
    root = Path(cfg.image_path)
    imgs = sorted([p for p in root.glob("*/*.img.nii.gz")]) or sorted(
        [p for p in root.glob("*/*.nii.gz") if "label" not in p.name])
    if not imgs: raise FileNotFoundError(f"No images in {root}")

    masks = None
    if cfg.mask_path or cfg.mask_suffix:
        suffix = cfg.mask_suffix or ".label.nii.gz"
        masks = [p.parent / f"{p.name.split('.img')[0]}{suffix}" for p in imgs]
        if not all(m.exists() for m in masks):
            if cfg.get('strict_matching', True): raise ValueError("Missing masks!")
            logger.warning("Masks missing, evaluation disabled.");
            masks = None

    return cfg, imgs, masks, root.parent.name


@hydra.main(config_path="../configs", config_name="inference/tem_infer", version_base="1.3.2")
def main(cfg):
    cfg, all_imgs, all_masks, ds_name = resolve_paths(cfg)
    gpus = list(cfg.gpus) if cfg.get("gpus") else [0]

    # 切分数据
    splits = lambda l, n: [l[i:i + math.ceil(len(l) / n)] for i in range(0, len(l), math.ceil(len(l) / n))]
    chunks_i, chunks_m = splits(all_imgs, len(gpus)), splits(all_masks, len(gpus)) if all_masks else [None] * len(gpus)

    logger.info(f"🚀 Start: {len(all_imgs)} files | {len(gpus)} GPUs | Dataset: {ds_name}")
    with mp.Manager() as manager:
        ret_dict = manager.dict()
        procs = [mp.Process(target=worker, args=(i, gpus[i], chunks_i[i], chunks_m[i], cfg, ret_dict)) for i in
                 range(len(chunks_i))]
        [p.start() for p in procs];
        [p.join() for p in procs]

        final_res = {k: v for d in ret_dict.values() for k, v in d.items()}
        if final_res:
            means = calculate_mean_metrics(list(final_res.values()), round_to=cfg.round_to)

            # 【修改点】控制台打印也按优先级排序
            priority = ['dice', 'cldice', 'clDice', 'nsd', 'asd']
            sorted_mean_keys = sorted(means.keys(), key=lambda x: (priority.index(x) if x in priority else 99, x))

            logger.info("\n" + "\n".join([f"Mean {k:<15}: {means[k]:.4f}" for k in sorted_mean_keys]))
            save_report(final_res, means, cfg, ds_name)
        else:
            logger.info("No metrics calculated.")


if __name__ == "__main__":
    main()