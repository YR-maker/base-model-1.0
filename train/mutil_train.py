import logging
import sys
import warnings
import os  # 确保导入 os
import numpy as np
from pathlib import Path
from monai.data import CacheDataset # 保留引用，虽然下面被自定义类替代，但保持兼容性

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
    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF  # 即 4294967295
except ImportError:
    pass
# ==========================================

import hydra
import torch
import torch.utils
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, Subset

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from utils.dataset import UnionDataset
from utils.evaluation import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# 在main函数之前定义辅助函数 (保持不变)
def _log_validation_details(phase, trainer, pl_module, dataset_name):
    """记录验证详细结果"""
    # 仅在主进程打印
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
    # 仅在主进程打印
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

    # 构建运行名称，包含关键实验信息
    dataset_name = list(cfg.data.keys())[0]  # 获取数据集名称
    # 1. 获取完整路径字符串
    full_data_path = cfg.data[dataset_name].path

    # 2. 使用 Path 对象提取最后一个文件夹名
    last_folder_name = os.path.basename(os.path.normpath(full_data_path))

    run_name = f'{cfg.loss_name}_{cfg.num_shots}shot_{last_folder_name}'

    # 强制设置为离线模式
    cfg.offline = True

    # ---------------------------------------------------------
    # 设置日志保存的绝对路径 (仅 Rank 0 创建目录)
    # ---------------------------------------------------------
    save_root_dir = "/home/yangrui/Project/Base-models/local_results/doc/" + cfg.data_name
    if global_rank == 0:
        os.makedirs(save_root_dir, exist_ok=True)  # 确保目录存在
        logger.info(f"📂 日志存储路径已设置为: {save_root_dir}")

    # 初始化Weights & Biases日志记录器（离线模式）
    wnb_logger = WandbLogger(
        save_dir=save_root_dir,
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        offline=True,
        mode="offline"
    )

    # 同时添加CSV日志记录器
    csv_logger = CSVLogger(
        save_dir=save_root_dir,
        name=run_name,
        version="version_0"
    )

    # ---------------------------------------------------------
    # 【核心修改 1】提前获取设备信息并计算动态步数
    # ---------------------------------------------------------
    # 确定使用的 GPU 列表和数量
    target_devices = cfg.devices
    num_devices = len(target_devices)

    # 读取 yaml 中的 max_steps (10000) 作为“基准总计算量”
    base_total_steps = cfg.trainer.lightning_trainer.max_steps

    # 逻辑：GPU 越多，单卡步数越少，保持总 Batch 量级一致
    # 4卡: 10000/4 = 2500 步; 2卡: 10000/2 = 5000 步
    actual_max_steps = int(base_total_steps // num_devices)

    # 逻辑：保持评估密度一致。总步数的 1/25 进行一次评估 (全程评估约25次)
    # 4卡: 2500/25 = 100步; 2卡: 5000/25 = 200步
    actual_val_interval = int(actual_max_steps // 25)

    if global_rank == 0:
        logger.info("=" * 40)
        logger.info(f"🧮 动态训练策略调整 (GPU数量: {num_devices})")
        logger.info("=" * 40)
        logger.info(f"   - YAML基准步数: {base_total_steps}")
        logger.info(f"   - 实际训练步数 (max_steps): {actual_max_steps}")
        logger.info(f"   - 评估间隔 (val_check_interval): {actual_val_interval} steps")
        logger.info("=" * 40)

    # 设置训练回调函数
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_DiceMetric"

    # 自定义回调函数，用于打印验证结果 (ValidationResultCallback) - 保持不变
    class ValidationResultCallback(LearningRateMonitor):
        def on_validation_end(self, trainer, pl_module):
            if trainer.global_rank != 0:
                return

            current_metrics = trainer.callback_metrics
            dice_score = current_metrics.get(f"{dataset_name}_val_dice", None)
            val_dice_metric = current_metrics.get("val_DiceMetric", None)
            val_loss = current_metrics.get(f"{dataset_name}_val_loss", None)

            current_step = trainer.global_step
            current_epoch = trainer.current_epoch

            logger.info("=" * 60)
            logger.info("📊 验证结果报告")
            logger.info("=" * 60)
            logger.info(f"🏃‍♂️ 当前训练进度: Epoch {current_epoch} | Step {current_step}")
            logger.info(f"🎯 数据集: {dataset_name}")

            if dice_score is not None:
                logger.info(f"✅ {dataset_name} Dice分数: {dice_score:.4f}")
            if val_dice_metric is not None:
                logger.info(f"🏆 验证Dice指标: {val_dice_metric:.4f}")
            if val_loss is not None:
                logger.info(f"📉 验证损失值: {val_loss:.4f}")

            if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback is not None:
                best_dice = trainer.checkpoint_callback.best_model_score
                if best_dice is not None:
                    logger.info(f"⭐ 历史最佳Dice: {best_dice:.4f}")
                    if val_dice_metric is not None:
                        improvement = val_dice_metric - best_dice
                        if improvement > 0:
                            logger.info(f"🚀 相比最佳提升: +{improvement:.4f}")
                        else:
                            logger.info(f"📌 距离最佳相差: {improvement:.4f}")

            logger.info("=" * 60)

    # ---------------------------------------------------------
    # 【核心修改 2】权重命名中加入 GPU 数量
    # ---------------------------------------------------------
    # filename 格式示例: step=2499_val_DiceMetric=0.85_4GPUs.ckpt
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.data_name + "/" + last_folder_name + "/" + run_name,
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        # 修改这里：在文件名末尾添加 _{num_devices}GPUs
        filename="{step}_{" + monitor_metric + ":.2f}_" + f"{num_devices}GPUs",
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    validation_callback = ValidationResultCallback()

    # 实例化 Trainer
    trainer_cls = hydra.utils.instantiate(cfg.trainer.lightning_trainer)

    # ---------------------------------------------------------
    # 【核心修改 3】使用动态计算的参数覆盖配置
    # ---------------------------------------------------------
    trainer_additional_kwargs = {
        "logger": [wnb_logger, csv_logger],
        "callbacks": [lr_monitor, checkpoint_callback, validation_callback],

        # 动态覆盖 yaml 中的配置
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
    # 【数据采样器调整】适配 UnionDataset 格式 (核心修复部分)
    # ---------------------------------------------------------

    # 0. 定义一个内部 Dataset 类
    # 作用：1. 存储在内存中的 List 数据; 2. 像 UnionDataset 一样返回 (Tuple) 而不是 (Dict)
    class FewShotInMemoryDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, transform):
            self.data = data_list
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            # 1. 应用 Transforms (MONAI Transforms 输入 Dict，输出 Dict)
            transformed = self.transform(item)

            # 2. 【关键修复】强制解包为 Tuple，模拟 UnionDataset 的行为
            # 必须返回 (Image, Mask) 的值，而不是 keys (字符串)
            return transformed['Image'], transformed['Mask'] > 0

    # 1. 实例化原始 Dataset 获取配置信息 (Reader, Transforms 等)
    raw_train_dataset = UnionDataset(cfg.data, "train", finetune=True)

    # 获取内部第一个数据集的信息
    dataset_info = raw_train_dataset.datasets[0]
    data_paths = dataset_info["paths"]
    reader = dataset_info["reader"]
    # 注意：UnionDataset 没有 .transform 属性，变换存储在 dataset_info 字典中
    data_transform = dataset_info["transforms"]

    subset_data_list = []

    # 2. 手动预加载前 num_shots 个样本的数据 (Image & Mask)
    # 必须在这里加载，因为 UnionDataset 的 transforms 期望输入是 Array 而不是 Path
    shots_to_load = min(cfg.num_shots, len(data_paths))

    if global_rank == 0:
        logger.info(f"🚀 正在将 {shots_to_load} 个 Few-Shot 样本手动加载到内存缓存中...")

    # 使用简单的循环读取数据
    for i in range(shots_to_load):
        sample_path = data_paths[i]

        # 复用 dataset.py 中的文件查找逻辑
        img_path = [p for p in sample_path.iterdir() if 'img' in p.name][0]
        mask_path = [p for p in sample_path.iterdir() if 'label' in p.name][0]

        # 复用 dataset.py 中的读取逻辑 (读取为 Numpy Array)
        img = reader.read_images(str(img_path))[0].astype(np.float32)
        mask = reader.read_images(str(mask_path))[0].astype(bool)

        # 构建符合 Transforms 预期的字典 (Keys 必须匹配 dataset.py 中的定义)
        subset_data_list.append({'Image': img, 'Mask': mask})

    if global_rank == 0:
        logger.info(f"✅ 成功加载 {len(subset_data_list)} 个样本到内存")

    # 3. 使用自定义的 Dataset 替代 CacheDataset
    # 这确保了 __getitem__ 返回的是 tuple(tensor, tensor) 而不是 dict
    train_dataset = FewShotInMemoryDataset(
        data_list=subset_data_list,
        transform=data_transform
    )

    # ---------------------------------------------------------
    # 采样器配置
    # ---------------------------------------------------------
    # 计算每张卡需要跑的样本数，保持总 Epoch 规模不变
    total_samples_per_epoch = int(1e5)
    samples_per_gpu = total_samples_per_epoch // num_devices

    if global_rank == 0:
        logger.info(f"Multi-GPU Config: {num_devices} GPUs ({target_devices})")
        logger.info(
            f"Sampler: {samples_per_gpu} samples per GPU (Total effective epoch size: {samples_per_gpu * num_devices})")

    # 使用随机采样器并进行重复采样
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=samples_per_gpu)

    train_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=train_dataset,
        sampler=random_sampler,
        # num_workers=4
    )

    # 验证和测试数据集 (保持不变)
    val_dataset = UnionDataset(cfg.data, "val", finetune=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
    if global_rank == 0: logger.info(f"Val dataset size: {len(val_dataset)}")

    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)
    if global_rank == 0: logger.info(f"Test dataset size: {len(test_dataset)}")

    # 初始化模型 (保持不变)
    model = hydra.utils.instantiate(cfg.model)

    # 加载预训练权重 (保持不变)
    if cfg.path_to_chkpt is not None:
        try:
            chkpt = torch.load(cfg.path_to_chkpt, map_location='cpu', weights_only=True)
        except:
            chkpt = torch.load(cfg.path_to_chkpt, map_location='cpu', weights_only=False)

        if isinstance(chkpt, dict):
            model_chkpt = chkpt.get('state_dict', chkpt.get('model_state_dict', chkpt.get('models', chkpt)))
        else:
            model_chkpt = chkpt

        if isinstance(model_chkpt, dict) and any(k.startswith('models.') for k in model_chkpt.keys()):
            from collections import OrderedDict
            model_chkpt = OrderedDict([(k.replace('models.', '', 1) if k.startswith('models.') else k, v)
                                       for k, v in model_chkpt.items()])

        model.load_state_dict(model_chkpt, strict=False)
        if global_rank == 0:
            logger.info(f"Loaded pretrained weights from {cfg.path_to_chkpt}")

    # 初始化Lightning模块 (保持不变)
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=evaluator,
        dataset_name=dataset_name
    )

    # 训练流程 (保持不变)
    if not cfg.offline:
        if global_rank == 0:
            wnb_logger.watch(model, log="all", log_freq=20)
    else:
        if global_rank == 0:
            logger.info("离线模式：跳过模型参数监控")

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
    # 设置标准输出缓冲，确保日志实时显示
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    # 设置环境变量，防止wandb询问
    import os

    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"

    main()


