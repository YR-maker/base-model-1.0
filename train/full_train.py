import logging, sys, warnings, os, hydra, torch
from pathlib import Path
from omegaconf import OmegaConf

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

# 引入支持 repeats 的 Dataset
from utils.full_train_dataset import UnionDataset
from utils.evaluation import Evaluator

# --- 环境设置 ---
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 屏蔽警告与冗余日志
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("monai").setLevel(logging.ERROR)

try:
    import monai.transforms.transform

    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF
except ImportError:
    pass

logger = logging.getLogger(__name__)


# --- 自定义组件 ---

class CleanCSVLogger(CSVLogger):
    """不保存 hparams.yaml 的清爽版 CSVLogger"""

    def log_hyperparams(self, params):
        pass


class LogCallback(LearningRateMonitor):
    """中文日志监控回调"""

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0: return
        m = trainer.callback_metrics
        d_name, epoch = pl_module.dataset_name, trainer.current_epoch

        score = m.get(f"{d_name}_val_dice") or m.get("val_DiceMetric")
        loss = m.get(f"{d_name}_val_loss") or m.get("val_loss")

        # 获取历史最佳 (注意：这里获取的是 Checkpoint callback 中记录的值)
        # 如果 Checkpoint callback 尚未更新，这里就是上一轮的最佳
        best = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None

        logger.info(f"{'=' * 30} Epoch {epoch} {'=' * 30}")
        if score: logger.info(f"✅ 验证 Dice: {score:.4f}")
        if loss:  logger.info(f"📉 验证 Loss: {loss:.4f}")

        if best and score:
            diff = float(score) - float(best)
            # 【修改点】优化显示文案，显示具体差值
            if diff > 1e-6:
                icon = f"🚀 新纪录! (+{diff:.4f})"  # 显示提升幅度
            elif diff > -1e-6:
                icon = "⚖️  持平"
            else:
                icon = f"🔙 差距: {diff:.4f}"  # 显示落后幅度(负数)

            logger.info(f"⭐ 历史最佳: {best:.4f} | {icon}")
        logger.info("=" * 67)


def safe_load_weights(model, ckpt_path, rank=0):
    """智能权重加载"""
    if not ckpt_path: return
    if rank == 0: logger.info(f"🔄 正在加载预训练权重: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    state = {k.replace('model.', '').replace('models.', '').replace('net.', ''): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError:
        if rank == 0: logger.warning("⚠️ 检测到层结构不匹配，正在智能过滤不兼容的层...")
        curr = model.state_dict()
        filtered_state = {k: v for k, v in state.items() if k in curr and v.shape == curr[k].shape}
        model.load_state_dict(filtered_state, strict=False)

    if rank == 0: logger.info("✅ 权重加载完成")


# --- 主程序 ---
@hydra.main(config_path="../configs", config_name="train/full_train", version_base="1.3.2")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    rank = int(os.environ.get("RANK", 0))

    d_name = list(cfg.data.keys())[0]
    run_name = f'FullTrain_{cfg.loss_name}_{os.path.basename(os.path.normpath(cfg.data[d_name].path))}'

    base_doc_path = f"../local_results/doc/{cfg.data_name}"
    experiment_dir = os.path.join(base_doc_path, run_name)
    if rank == 0: os.makedirs(experiment_dir, exist_ok=True)

    loggers = [
        WandbLogger(save_dir=experiment_dir, name=run_name, config=OmegaConf.to_container(cfg), offline=True,
                    mode="offline"),
        CleanCSVLogger(save_dir=experiment_dir, name="", version="")
    ]

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{cfg.chkpt_folder}/{cfg.data_name}/{os.path.basename(os.path.normpath(cfg.data[d_name].path))}/{run_name}",
        monitor="val_DiceMetric",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="epoch:{epoch:d}-dice:{val_DiceMetric:.2f}",
        auto_insert_metric_name=False
    )
    ckpt_cb.CHECKPOINT_EQUALS_CHAR = "="
    ckpt_cb.CHECKPOINT_NAME_LAST = f"{run_name}_last"

    # --- 数据加载配置 ---
    # 【核心修改】从配置文件读取 repeats，默认为 1
    # 这样你只需要在 yaml 里改 repeats: 8 即可
    repeats = cfg.get("repeats", 1)

    train_dataset = UnionDataset(cfg.data, "train", finetune=True, repeats=repeats)
    val_dataset = UnionDataset(cfg.data, "val", finetune=True, repeats=1)
    test_dataset = UnionDataset(cfg.data, "test", finetune=True, repeats=1)

    if rank == 0:
        logger.info(f"📊 [数据集概览] 全量训练模式")
        logger.info(f"   - 原始样本数: {len(train_dataset) // repeats}")
        logger.info(f"   - 采样倍率 (repeats): {repeats} (每个Epoch对每张图裁 {repeats} 次)")
        logger.info(f"   - Train (Virtual): {len(train_dataset)} 样本")
        logger.info(f"   - Val: {len(val_dataset)} 样本")

    train_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1, shuffle=False)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1, shuffle=False)

    model = hydra.utils.instantiate(cfg.model)
    safe_load_weights(model, cfg.path_to_chkpt, rank)

    pl_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=Evaluator(),
        dataset_name=d_name
    )

    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer,
                                      logger=loggers,
                                      callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
                                      devices=cfg.devices,
                                      accelerator="gpu",
                                      strategy="ddp",
                                      sync_batchnorm=True,
                                      enable_model_summary=False,
                                      )()

    if rank == 0:
        logger.info(f"🚀 开始全量训练")
        logger.info(f"   - 总轮数: {trainer.max_epochs}")
        logger.info(f"   - 评估频率: 每 {trainer.check_val_every_n_epoch} Epochs")

    trainer.fit(pl_module, train_loader, val_loader)

    if rank == 0: logger.info("✅ 训练完成，开始最终测试...")
    trainer.test(pl_module, test_loader, ckpt_path="best")

    if rank == 0:
        res = trainer.callback_metrics
        logger.info(f"\n🏆 最终测试 Dice: {res.get('test_DiceMetric', 0):.4f}")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()