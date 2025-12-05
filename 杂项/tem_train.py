import logging
import sys
import warnings

# ==========================================
# 【关键修复】MONAI 与 NumPy 版本兼容性修复
# 必须放在 from utils.dataset import UnionDataset 之前
try:
    import monai.transforms.transform
    # 强制修改 MONAI 内部的 MAX_SEED，防止 NumPy 报错 (OverflowError)
    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF # 即 4294967295
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


@hydra.main(config_path="../configs", config_name="tem_train", version_base="1.3.2")
def main(cfg):
    """
    模型的微调主函数
    - 零样本(zero-shot): num_shots=0，直接测试预训练模型
    - 单样本(one-shot): num_shots=1，使用1个样本微调
    - 少样本(few-shot): num_shots=3，使用3个样本微调
    """

    # 设置随机种子确保实验可复现性
    seed_everything(cfg.seed, True)
    # 设置矩阵乘法精度平衡速度和精度
    torch.set_float32_matmul_precision("medium")

    # 构建运行名称，包含关键实验信息
    dataset_name = list(cfg.data.keys())[0]  # 获取数据集名称
    run_name = f'{cfg.num_shots}shot_{dataset_name}'

    # 强制设置为离线模式，避免询问上传网络
    cfg.offline = True

    # ---------------------------------------------------------
    # 【修改点】设置日志保存的绝对路径
    # ---------------------------------------------------------
    save_root_dir = "/home/yangrui/Project/Base-models/local_results/doc"
    os.makedirs(save_root_dir, exist_ok=True) # 确保目录存在
    logger.info(f"📂 日志存储路径已设置为: {save_root_dir}")

    # 初始化Weights & Biases日志记录器（离线模式）
    wnb_logger = WandbLogger(
        save_dir=save_root_dir,     # <--- 修改：指定wandb保存路径
        project=cfg.wandb_project,  # 项目名称
        name=run_name,              # 运行名称
        config=OmegaConf.to_container(cfg),  # 记录完整配置
        offline=True,               # 强制离线模式，不询问上传
        mode="offline"              # 明确设置为离线模式
    )

    # 同时添加CSV日志记录器，确保日志在本地存储
    csv_logger = CSVLogger(
        save_dir=save_root_dir,     # <--- 修改：指定CSV日志保存路径
        name=run_name,              # 运行名称
        version="version_0"         # 版本号
    )

    # 设置训练回调函数
    lr_monitor = LearningRateMonitor()  # 学习率监控
    monitor_metric = "val_DiceMetric"  # 监控指标（Dice系数）

    # 自定义回调函数，用于打印验证结果
    class ValidationResultCallback(LearningRateMonitor):
        def on_validation_end(self, trainer, pl_module):
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
        dirpath=cfg.chkpt_folder + "/" + cfg.wandb_project + "/" + run_name,  # 保存路径
        monitor=monitor_metric,  # 监控的指标
        save_top_k=1,  # 只保存最好的1个模型
        mode="max",  # 指标越大越好
        filename="{step}_{" + monitor_metric + ":.2f}",  # 文件名格式
        auto_insert_metric_name=True,  # 自动插入指标名
        save_last=True  # 同时保存最后一个epoch的模型
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    # 初始化验证结果回调
    validation_callback = ValidationResultCallback()

    # 初始化PyTorch Lightning训练器
    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": [wnb_logger, csv_logger],  # 使用多个日志记录器
        "callbacks": [lr_monitor, checkpoint_callback, validation_callback],  # 回调函数
        "devices": cfg.devices  # 训练设备
    }
    trainer = trainer(**trainer_additional_kwargs)

    # 初始化数据加载器 - 关键部分对应论文中的实验设置
    # 训练数据集：使用UnionDataset并限制样本数量
    train_dataset = UnionDataset(cfg.data, "fine-tuning", finetune=True)
    train_dataset = Subset(train_dataset, range(cfg.num_shots))  # 限制样本数量

    # 使用随机采样器并进行重复采样，模拟论文中的少样本设置
    # 随机采样两万次，每次训练为两万个样本，batch size为4，共训练2500轮次，每500次进行一次验证，共验证25次
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=int(1e4))
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset, sampler=random_sampler)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    # 验证数据集
    val_dataset = UnionDataset(cfg.data, "val", finetune=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # 测试数据集
    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # 初始化模型
    model = hydra.utils.instantiate(cfg.model)

    # 加载预训练权重（如果指定了检查点路径）
    if cfg.path_to_chkpt is not None:
        try:
            chkpt = torch.load(cfg.path_to_chkpt, map_location=f'cuda:{cfg.devices[0]}', weights_only=True)
        except:
            chkpt = torch.load(cfg.path_to_chkpt, map_location=f'cuda:{cfg.devices[0]}', weights_only=False)

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
        logger.info(f"Loaded pretrained weights from {cfg.path_to_chkpt}")


    # 初始化Lightning模块 - 封装训练逻辑
    evaluator = Evaluator()  # 评估器，用于计算Dice和clDice指标
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=evaluator,
        dataset_name=dataset_name
    )

    # 只在在线模式下监控模型参数（离线模式下跳过）
    if not cfg.offline:
        wnb_logger.watch(model, log="all", log_freq=20)  # 监控模型参数
    else:
        logger.info("离线模式：跳过模型参数监控")

        # 根据样本数量选择不同的实验模式
        if cfg.num_shots == 0:
            # 零样本评估：直接测试预训练模型
            logger.info("Starting zero-shot evaluation")
            trainer.test(lightning_module, test_loader)
        else:
            # 少样本微调：验证→训练→测试完整流程
            logger.info("Starting training")

            # 初始验证并打印结果
            logger.info("🔍 进行初始验证...")
            initial_val_results = trainer.validate(lightning_module, val_loader)
            # 记录详细日志
            _log_validation_details("初始验证", trainer, lightning_module, dataset_name)

            # 开始训练
            logger.info("🚀 开始模型训练...")
            trainer.fit(lightning_module, train_loader, val_loader)

            logger.info("Finished training")

            # 最终测试并打印结果
            logger.info("🧪 进行最终测试...")
            trainer.test(lightning_module, test_loader, ckpt_path="best")

            # 打印测试结果摘要
            _log_test_summary(trainer, lightning_module, dataset_name)

        # 记录实验完成信息
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