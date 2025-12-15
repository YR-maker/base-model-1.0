import logging
import sys
import warnings
import os

#该训练是只使用了连通性那篇论文的损失函数

from pathlib import Path

# 获取当前脚本的绝对路径
current_file_path = Path(__file__).resolve()
# 获取项目根目录 (即 fine-tuning 文件夹的上一级)
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


# 在main函数之前定义辅助函数
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


@hydra.main(config_path="../configs", config_name="con_loss_train", version_base="1.3.2")
def main(cfg):
    """
    模型的微调主函数 (已适配多卡DDP训练)
    """

    # 设置随机种子确保实验可复现性
    seed_everything(cfg.seed, True)
    # 设置矩阵乘法精度平衡速度和精度
    torch.set_float32_matmul_precision("medium")

    # 获取当前进程的全局 rank，用于控制日志打印
    # Lightning 初始化前可以通过环境变量获取，默认为 0
    global_rank = int(os.environ.get("RANK", 0))

    # 构建运行名称，包含关键实验信息
    dataset_name = list(cfg.data.keys())[0]  # 获取数据集名称
    # === 【新增代码】获取路径最后一部分 ===
    # 1. 获取完整路径字符串 (例如: /home/yangrui/Project/Base-model/input/imageCAS)
    full_data_path = cfg.data[dataset_name].path

    # 2. 使用 Path 对象提取最后一个文件夹名 (例如: imageCAS)
    # Path(路径).name 会自动获取路径的最后一部分
    last_folder_name = os.path.basename(os.path.normpath(full_data_path))

    run_name = f'{cfg.loss_name}_{cfg.num_shots}shot_{last_folder_name}'

    # 强制设置为离线模式
    cfg.offline = True

    # ---------------------------------------------------------
    # 【修改点】设置日志保存的绝对路径 (仅 Rank 0 创建目录)
    # ---------------------------------------------------------
    save_root_dir = "/home/yangrui/Project/Base-model/local_results/doc"
    if global_rank == 0:
        os.makedirs(save_root_dir, exist_ok=True)  # 确保目录存在
        logger.info(f"📂 日志存储路径已设置为: {save_root_dir}")

    # 初始化Weights & Biases日志记录器（离线模式）
    # Lightning 会自动处理 Logger 的多进程逻辑，无需手动限制 rank
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

    # 设置训练回调函数
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_DiceMetric"

    # 自定义回调函数，用于打印验证结果
    class ValidationResultCallback(LearningRateMonitor):
        def on_validation_end(self, trainer, pl_module):
            # 仅在主进程打印日志
            if trainer.global_rank != 0:
                return

            # 获取当前验证指标
            current_metrics = trainer.callback_metrics
            dice_score = current_metrics.get(f"{dataset_name}_val_dice", None)
            val_dice_metric = current_metrics.get("val_DiceMetric", None)
            val_loss = current_metrics.get(f"{dataset_name}_val_loss", None)

            # 获取当前训练步数和epoch
            current_step = trainer.global_step
            current_epoch = trainer.current_epoch

            # 打印详细的验证结果
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

            # 打印最佳指标对比
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

    # 模型检查点回调 - 保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.data_name + "/" + last_folder_name + "/" + run_name,
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename="{step}_{" + monitor_metric + ":.2f}",
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    # 初始化验证结果回调
    validation_callback = ValidationResultCallback()

    # ---------------------------------------------------------
    # 【修改点】配置多卡训练参数
    # ---------------------------------------------------------
    # 确定使用的 GPU 数量
    num_devices = cfg.devices_num  # 强制设置为 2 卡，或者读取 len(cfg.devices)

    # 实例化 Trainer
    trainer_cls = hydra.utils.instantiate(cfg.trainer.lightning_trainer)

    # 覆盖参数以启用 DDP 多卡训练
    trainer_additional_kwargs = {
        "logger": [wnb_logger, csv_logger],
        "callbacks": [lr_monitor, checkpoint_callback, validation_callback],
        "devices": num_devices,  # 使用4张显卡
        "accelerator": "gpu",  # 加速器类型
        "strategy": "ddp",  # 分布式数据并行策略
        "sync_batchnorm": True,  # 【重要】多卡同步BatchNorm，对分割任务至关重要
        "use_distributed_sampler": False  # 【重要】禁用默认采样器，使用自定义RandomSampler
    }
    # 如果 cfg 中已经实例化了 trainer 对象，这里可能需要调整写法
    # 通常 hydra instantiate 返回的是对象，这里假设它返回的是 partial 或者我们重新构造
    # 为了保险，我们直接用 Trainer 类封装参数，或者沿用原逻辑覆盖
    # 原逻辑是: trainer = hydra... -> trainer(**kwargs)
    # 这里的 cfg.trainer.lightning_trainer 应该是一个 _partial_: True 的配置
    trainer = trainer_cls(**trainer_additional_kwargs)

    # ---------------------------------------------------------
    # 【修改点】调整数据采样器以适配多卡
    # ---------------------------------------------------------
    train_dataset = UnionDataset(cfg.data, "train", finetune=True)
    train_dataset = Subset(train_dataset, range(cfg.num_shots))

    # 计算每张卡需要跑的样本数，保持总 Epoch 规模不变 (约10000)
    total_samples_per_epoch = int(1e5)
    samples_per_gpu = total_samples_per_epoch // num_devices

    if global_rank == 0:
        logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")
        logger.info(f"Multi-GPU Config: {num_devices} GPUs")
        logger.info(
            f"Sampler: {samples_per_gpu} samples per GPU (Total effective epoch size: {samples_per_gpu * num_devices})")

    # 使用随机采样器并进行重复采样
    # 注意：在DDP模式下，如果不使用DistributedSampler，每张卡都会独立进行RandomSampling
    # 因为我们是 replacement=True 且样本极少，这种独立随机是完全可以接受的
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=samples_per_gpu)

    train_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=train_dataset,
        sampler=random_sampler,
        # 建议在多卡训练时适当增加 num_workers
        # num_workers=4
    )

    # 验证和测试数据集
    val_dataset = UnionDataset(cfg.data, "val", finetune=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
    if global_rank == 0: logger.info(f"Val dataset size: {len(val_dataset)}")

    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)
    if global_rank == 0: logger.info(f"Test dataset size: {len(test_dataset)}")

    # 初始化模型
    model = hydra.utils.instantiate(cfg.model)

    # 加载预训练权重（如果指定了检查点路径）
    if cfg.path_to_chkpt is not None:
        # 【修改】加载权重时映射到 CPU，避免多进程占用导致的问题，随后 Lightning 会自动转到 GPU
        try:
            chkpt = torch.load(cfg.path_to_chkpt, map_location='cpu', weights_only=True)
        except:
            chkpt = torch.load(cfg.path_to_chkpt, map_location='cpu', weights_only=False)

        # 处理状态字典
        if isinstance(chkpt, dict):
            model_chkpt = chkpt.get('state_dict', chkpt.get('model_state_dict', chkpt.get('models', chkpt)))
        else:
            model_chkpt = chkpt

        # 移除"models."前缀（如果需要）
        if isinstance(model_chkpt, dict) and any(k.startswith('models.') for k in model_chkpt.keys()):
            from collections import OrderedDict
            model_chkpt = OrderedDict([(k.replace('models.', '', 1) if k.startswith('models.') else k, v)
                                       for k, v in model_chkpt.items()])

        model.load_state_dict(model_chkpt, strict=False)
        if global_rank == 0:
            logger.info(f"Loaded pretrained weights from {cfg.path_to_chkpt}")

    # 初始化Lightning模块
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=evaluator,
        dataset_name=dataset_name
    )

    # 训练流程
    if not cfg.offline:
        if global_rank == 0:
            wnb_logger.watch(model, log="all", log_freq=20)
    else:
        if global_rank == 0:
            logger.info("离线模式：跳过模型参数监控")

        # 根据样本数量选择不同的实验模式
        if cfg.num_shots == 0:
            if global_rank == 0: logger.info("Starting zero-shot evaluation")
            trainer.test(lightning_module, test_loader)
        else:
            if global_rank == 0:
                logger.info("Starting training")
                logger.info("🔍 进行初始验证...")

            # 初始验证
            trainer.validate(lightning_module, val_loader)

            if global_rank == 0:
                _log_validation_details("初始验证", trainer, lightning_module, dataset_name)
                logger.info("🚀 开始模型训练...")

            # 开始训练
            trainer.fit(lightning_module, train_loader, val_loader)

            if global_rank == 0:
                logger.info("Finished training")
                logger.info("🧪 进行最终测试...")

            # 最终测试
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