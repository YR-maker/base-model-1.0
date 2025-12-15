import logging
import sys
import warnings
import os
import numpy as np
from pathlib import Path

# 获取当前脚本的绝对路径
current_file_path = Path(__file__).resolve()
# 获取项目根目录 (即 train 文件夹的上一级)
project_root = current_file_path.parent.parent
# 将项目根目录添加到 python 搜索路径中
sys.path.append(str(project_root))

# ==========================================
# 【关键修复】MONAI 与 NumPy 版本兼容性修复
# 必须放在 from utils.dataset import UnionDataset 之前
try:
    import monai.transforms.transform

    # 强制修改 MONAI 内部的 MAX_SEED，防止 NumPy 报错 (OverflowError)
    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF
except ImportError:
    pass
# ==========================================

import hydra
import torch
import torch.utils
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from utils.dataset import UnionDataset
from utils.evaluation import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _log_validation_details(phase, trainer, pl_module, dataset_name):
    """记录验证详细结果"""
    if trainer.global_rank != 0:
        return

    current_metrics = trainer.callback_metrics
    dice_score = current_metrics.get(f"{dataset_name}_val_dice", None)
    val_dice_metric = current_metrics.get("val_DiceMetric", None)

    logger.info("📈 " + "=" * 50)
    logger.info(f"📋 {phase}结果摘要")
    logger.info("📈 " + "=" * 50)
    logger.info(f"📊 数据集: {dataset_name}")
    if dice_score is not None:
        logger.info(f"🎯 数据集Dice: {dice_score:.4f}")
    if val_dice_metric is not None:
        logger.info(f"⭐ 综合Dice指标: {val_dice_metric:.4f}")
    logger.info("📈 " + "=" * 50)


def _log_test_summary(trainer, pl_module, dataset_name):
    """记录测试结果摘要"""
    if trainer.global_rank != 0:
        return

    current_metrics = trainer.callback_metrics
    test_dice = current_metrics.get(f"{dataset_name}_test_dice", None)
    test_dice_metric = current_metrics.get("test_DiceMetric", None)

    logger.info("🎉 " + "=" * 60)
    logger.info("🏆 最终测试结果报告")
    logger.info("🎉 " + "=" * 60)
    if test_dice is not None:
        logger.info(f"✅ 测试集Dice分数: {test_dice:.4f}")
    if test_dice_metric is not None:
        logger.info(f"🏅 最终Dice指标: {test_dice_metric:.4f}")
    logger.info("🎉 " + "=" * 60)


@hydra.main(config_path="../configs", config_name="mutil_train", version_base="1.3.2")
def main(cfg):
    """
    模型的微调主函数 (已适配多卡DDP训练及动态步数调整)
    """

    # 设置随机种子确保实验可复现性
    seed_everything(cfg.seed, True)
    # 设置矩阵乘法精度平衡速度和精度
    torch.set_float32_matmul_precision("medium")

    # 获取当前进程的全局 rank，用于控制日志打印
    global_rank = int(os.environ.get("RANK", 0))

    # 构建运行名称
    dataset_name = list(cfg.data.keys())[0]
    full_data_path = cfg.data[dataset_name].path
    last_folder_name = os.path.basename(os.path.normpath(full_data_path))
    run_name = f'{cfg.loss_name}_{cfg.num_shots}shot_{last_folder_name}'

    # 强制设置为离线模式
    cfg.offline = True

    # 设置日志保存路径
    save_root_dir = "/home/yangrui/Project/Base-model/local_results/doc/" + cfg.data_name
    if global_rank == 0:
        os.makedirs(save_root_dir, exist_ok=True)
        logger.info(f"📂 日志存储路径已设置为: {save_root_dir}")

    # 初始化日志记录器
    wnb_logger = WandbLogger(
        save_dir=save_root_dir,
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        offline=True,
        mode="offline"
    )

    csv_logger = CSVLogger(
        save_dir=save_root_dir,
        name=run_name,
        version="version_0"
    )

    # ---------------------------------------------------------
    # 动态计算步数策略
    # ---------------------------------------------------------
    target_devices = cfg.devices
    num_devices = len(target_devices)
    base_total_steps = cfg.trainer.lightning_trainer.max_steps

    # 动态调整 max_steps 和 val_check_interval
    actual_max_steps = int(base_total_steps // num_devices)
    actual_val_interval = int(actual_max_steps // 25)

    if global_rank == 0:
        logger.info("=" * 40)
        logger.info(f"🧮 动态训练策略调整 (GPU数量: {num_devices})")
        logger.info(f"   - YAML基准步数: {base_total_steps}")
        logger.info(f"   - 实际训练步数: {actual_max_steps}")
        logger.info(f"   - 评估间隔: {actual_val_interval} steps")
        logger.info("=" * 40)

    # 设置回调函数
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_DiceMetric"

    class ValidationResultCallback(LearningRateMonitor):
        def on_validation_end(self, trainer, pl_module):
            if trainer.global_rank != 0: return

            # 获取指标并打印
            current_metrics = trainer.callback_metrics
            # 注意：这里从 metrics 取出的可能是 Tensor，计算时最好转为 float
            dice_score = current_metrics.get(f"{dataset_name}_val_dice", None)
            val_dice_metric = current_metrics.get("val_DiceMetric", None)
            val_loss = current_metrics.get(f"{dataset_name}_val_loss", None)
            current_epoch = trainer.current_epoch

            logger.info("=" * 60)
            logger.info(f"📊 验证结果报告 (Epoch {current_epoch})")
            if dice_score is not None: logger.info(f"✅ {dataset_name} Dice: {dice_score:.4f}")
            if val_dice_metric is not None: logger.info(f"🏆 验证Dice指标: {val_dice_metric:.4f}")
            if val_loss is not None: logger.info(f"📉 验证损失值: {val_loss:.4f}")

            # --- 修改开始：添加历史最佳与差距计算 ---
            if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback is not None:
                best_dice = trainer.checkpoint_callback.best_model_score

                if best_dice is not None:
                    logger.info(f"⭐ 历史最佳Dice: {best_dice:.4f}")

                    # 只有当当前分数也存在时，才计算差距
                    if dice_score is not None:
                        current_val = float(dice_score)
                        best_val = float(best_dice)
                        diff = current_val - best_val

                        # 格式化输出：如果是正数加 '+' 号，且用不同图标表示
                        if diff > 0:
                            logger.info(f"🚀 新纪录! 提升: +{diff:.4f}")
                        elif diff == 0:
                            logger.info(f"⚖️  持平历史最佳")
                        else:
                            logger.info(f"🔙 距历史最佳: {diff:.4f}")
            # --- 修改结束 ---

            logger.info("=" * 60)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.data_name + "/" + last_folder_name + "/" + run_name,
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename="{step}_{" + monitor_metric + ":.2f}_" + f"{num_devices}GPUs",
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    validation_callback = ValidationResultCallback()

    # 实例化 Trainer
    trainer_cls = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": [wnb_logger, csv_logger],
        "callbacks": [lr_monitor, checkpoint_callback, validation_callback],
        "max_steps": actual_max_steps,
        "val_check_interval": actual_val_interval,
        "devices": target_devices,
        "accelerator": "gpu",
        "strategy": "ddp",
        "sync_batchnorm": True,
        "use_distributed_sampler": False
    }
    trainer = trainer_cls(**trainer_additional_kwargs)

    # ---------------------------------------------------------
    # 数据集定义 (内存加载类)
    # ---------------------------------------------------------
    class FewShotInMemoryDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, transform):
            self.data = data_list
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            # 实时应用变换
            transformed = self.transform(item)
            # 必须返回 (Image, Mask) 元组
            return transformed['Image'], transformed['Mask'] > 0

    # ---------------------------------------------------------
    # 1. 训练集 (Few-Shot): 加载到内存
    # ---------------------------------------------------------
    raw_train_dataset = UnionDataset(cfg.data, "train", finetune=True)
    dataset_info = raw_train_dataset.datasets[0]
    data_paths = dataset_info["paths"]
    reader = dataset_info["reader"]
    data_transform = dataset_info["transforms"]

    subset_data_list = []
    shots_to_load = min(cfg.num_shots, len(data_paths))

    if global_rank == 0:
        logger.info(f"🚀 正在加载 {shots_to_load} 个训练样本到内存...")

    for i in range(shots_to_load):
        sample_path = data_paths[i]
        img_path = [p for p in sample_path.iterdir() if 'img' in p.name][0]
        mask_path = [p for p in sample_path.iterdir() if 'label' in p.name][0]

        img = reader.read_images(str(img_path))[0].astype(np.float32)
        mask = reader.read_images(str(mask_path))[0].astype(bool)
        subset_data_list.append({'Image': img, 'Mask': mask})

    train_dataset = FewShotInMemoryDataset(data_list=subset_data_list, transform=data_transform)

    # 采样器配置
    total_samples_per_epoch = int(1e5)
    samples_per_gpu = total_samples_per_epoch // num_devices
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=samples_per_gpu)

    train_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=train_dataset,
        sampler=random_sampler
    )

    # ---------------------------------------------------------
    # 2. 验证集 (Validation): 加载到内存 (加速评估)
    # ---------------------------------------------------------
    def _load_split_to_memory(cfg, phase, global_rank):
        """通用辅助函数：将数据集加载到内存"""
        raw_dataset = UnionDataset(cfg.data, phase, finetune=True)
        if not raw_dataset.datasets or len(raw_dataset) == 0:
            return None

        d_info = raw_dataset.datasets[0]
        d_paths = d_info["paths"]
        d_reader = d_info["reader"]
        d_transform = d_info["transforms"]
        d_list = []

        if global_rank == 0:
            logger.info(f"🚀 正在将 {phase} 集 ({len(d_paths)} 样本) 加载到内存...")

        for s_path in d_paths:
            i_path = [p for p in s_path.iterdir() if 'img' in p.name][0]
            m_path = [p for p in s_path.iterdir() if 'label' in p.name][0]
            img = d_reader.read_images(str(i_path))[0].astype(np.float32)
            mask = d_reader.read_images(str(m_path))[0].astype(bool)
            d_list.append({'Image': img, 'Mask': mask})

        return FewShotInMemoryDataset(data_list=d_list, transform=d_transform)

    val_dataset = _load_split_to_memory(cfg, "val", global_rank)
    if val_dataset is not None:
        val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
        if global_rank == 0: logger.info(f"Val dataset size (In-Memory): {len(val_dataset)}")
    else:
        val_loader = None

    # ---------------------------------------------------------
    # 3. 测试集 (Test): 保持硬盘读取 (节省内存)
    # ---------------------------------------------------------
    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)
    if global_rank == 0: logger.info(f"Test dataset size (Disk-Based): {len(test_dataset)}")

    # 初始化模型
    model = hydra.utils.instantiate(cfg.model)

    # ---------------------------------------------------------
    # 权重加载逻辑
    # ---------------------------------------------------------
    if cfg.path_to_chkpt is not None:
        if global_rank == 0: logger.info(f"🔄 正在加载权重: {cfg.path_to_chkpt}")
        try:
            chkpt = torch.load(cfg.path_to_chkpt, map_location='cpu', weights_only=True)
        except (TypeError, Exception):
            chkpt = torch.load(cfg.path_to_chkpt, map_location='cpu', weights_only=False)

        if isinstance(chkpt, dict) and 'state_dict' in chkpt:
            model_chkpt = chkpt['state_dict']
            is_lightning = True
        else:
            model_chkpt = chkpt.get('state_dict', chkpt.get('model_state_dict', chkpt.get('models', chkpt)))
            is_lightning = False

        new_state_dict = {}
        for k, v in model_chkpt.items():
            new_key = k
            if is_lightning and k.startswith('model.'):
                new_key = k.replace('model.', '', 1)
            elif k.startswith('models.'):
                new_key = k.replace('models.', '', 1)
            new_state_dict[new_key] = v

        try:
            model.load_state_dict(new_state_dict, strict=False)
            if global_rank == 0: logger.info(f"✅ 成功加载权重")
        except RuntimeError as e:
            # 智能剔除不匹配的层
            if global_rank == 0: logger.warning(f"⚠️ 完整加载失败，尝试智能剔除...")
            current_model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in new_state_dict.items()
                             if k in current_model_dict and v.shape == current_model_dict[k].shape}
            model.load_state_dict(filtered_dict, strict=False)
            if global_rank == 0: logger.info("✅ 已加载匹配层")

    # 初始化 LightningModule
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=evaluator,
        dataset_name=dataset_name
    )

    # 训练流程
    if not cfg.offline:
        if global_rank == 0: wnb_logger.watch(model, log="all", log_freq=20)
    else:
        if global_rank == 0: logger.info("离线模式：跳过模型参数监控")

    if cfg.num_shots == 0:
        if global_rank == 0: logger.info("Starting zero-shot evaluation")
        trainer.test(lightning_module, test_loader)
    else:
        if global_rank == 0:
            logger.info("Starting training")
            logger.info("🔍 进行初始验证...")

        trainer.validate(lightning_module, val_loader)
        if global_rank == 0:
            _log_validation_details("初始验证", trainer, lightning_module, dataset_name)
            logger.info("🚀 开始模型训练...")

        trainer.fit(lightning_module, train_loader, val_loader)

        if global_rank == 0:
            logger.info("Finished training")
            logger.info("🧪 进行最终测试...")

        trainer.test(lightning_module, test_loader, ckpt_path="best")
        if global_rank == 0:
            _log_test_summary(trainer, lightning_module, dataset_name)
            logger.info(f"实验完成！日志保存在：{save_root_dir}")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()