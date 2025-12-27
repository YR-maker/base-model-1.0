import logging, sys, warnings, os, hydra, torch, numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, Dataset
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from utils.dataset import UnionDataset
from utils.evaluation import Evaluator

# --- 环境与兼容性设置 ---
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 【屏蔽警告】
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
os.environ["PYTHONWARNINGS"] = "ignore"

# 【屏蔽冗余控制台输出】
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("monai").setLevel(logging.ERROR)

try:
    import monai.transforms.transform

    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF
except ImportError:
    pass

logger = logging.getLogger(__name__)


# --- 辅助类与函数 ---

# 【新增】自定义 CSVLogger，用于禁止保存 hparams.yaml
class CleanCSVLogger(CSVLogger):
    """
    修改版的 CSVLogger：
    1. 不保存 hparams.yaml
    2. 仅保存 metrics.csv
    """

    def log_hyperparams(self, params):
        # 覆盖父类方法，什么都不做，从而禁止保存 hparams.yaml
        pass


class InMemoryDataset(Dataset):
    """内存数据集，提升小样本训练速度"""

    def __init__(self, data, transform): self.data, self.transform = data, transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        d = self.transform(self.data[idx])
        return d['Image'], d['Mask'] > 0


def safe_load_weights(model, ckpt_path, rank=0):
    """智能加载权重"""
    if not ckpt_path: return
    if rank == 0: logger.info(f"🔄 正在加载权重: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    state = {k.replace('model.', '').replace('models.', ''): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError:
        if rank == 0: logger.warning("⚠️ 检测到层结构不匹配，正在智能过滤...")
        curr = model.state_dict()
        state = {k: v for k, v in state.items() if k in curr and v.shape == curr[k].shape}
        model.load_state_dict(state, strict=False)
    if rank == 0: logger.info("✅ 权重加载成功。")


def get_loader(cfg, phase, rank=0, batch_size=1):
    raw_ds = UnionDataset(cfg.data, phase, finetune=True)
    if phase == "test":
        if rank == 0: logger.info(f"测试集: {len(raw_ds)}")
        return hydra.utils.instantiate(cfg.dataloader)(dataset=raw_ds, batch_size=batch_size)

    if not raw_ds.datasets: return None
    info = raw_ds.datasets[0]
    paths, reader, trans = info["paths"], info["reader"], info["transforms"]

    limit = min(cfg.num_shots, len(paths)) if phase == "train" else len(paths)
    if rank == 0: logger.info(f"🚀 正在将 {phase} 集 ({limit} 样本) 加载到内存...")

    data = []
    for p in paths[:limit]:
        img = reader.read_images(str(next(p.glob('*img*'))))[0].astype(np.float32)
        msk = reader.read_images(str(next(p.glob('*label*'))))[0].astype(bool)
        data.append({'Image': img, 'Mask': msk})

    mem_ds = InMemoryDataset(data, trans)

    if phase == "train":
        sampler = RandomSampler(mem_ds, replacement=True, num_samples=int(1e5) // len(cfg.devices))
        return hydra.utils.instantiate(cfg.dataloader)(dataset=mem_ds, sampler=sampler)
    else:
        return hydra.utils.instantiate(cfg.dataloader)(dataset=mem_ds, batch_size=1)


# --- 核心回调 ---
class LogCallback(LearningRateMonitor):
    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0: return
        m = trainer.callback_metrics
        d_name, epoch = pl_module.dataset_name, trainer.current_epoch
        score, loss = m.get(f"{d_name}_val_dice"), m.get(f"{d_name}_val_loss")
        best = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None

        logger.info(f"{'=' * 30} Epoch {epoch} {'=' * 30}")
        if score: logger.info(f"✅ {d_name} Dice: {score:.4f}")
        if loss:  logger.info(f"📉 验证 Loss: {loss:.4f}")

        if best and score:
            diff = float(score) - float(best)
            icon = "🚀 新纪录!" if diff > 0 else ("⚖️  持平" if diff == 0 else f"🔙 差距: {diff:.4f}")
            logger.info(f"⭐ 历史最佳: {best:.4f} | {icon}")
        logger.info("=" * 67)


# --- 主程序 ---
@hydra.main(config_path="../configs", config_name="train/mutil_train", version_base="1.3.2")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    rank = int(os.environ.get("RANK", 0))

    d_name = list(cfg.data.keys())[0]
    # 构建实验名称
    run_name = f'{cfg.loss_name}_{cfg.num_shots}shot_{os.path.basename(os.path.normpath(cfg.data[d_name].path))}'

    # 【修改点 1】构建完整的实验目录路径：.../doc/{数据集}/{实验名}
    base_doc_path = f"/home/yangrui/Project/Base-model/local_results/doc/{cfg.data_name}"
    experiment_dir = os.path.join(base_doc_path, run_name)

    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)

    steps = cfg.trainer.lightning_trainer.max_steps // len(cfg.devices)
    val_int = max(1, steps // 25)
    if rank == 0: logger.info(f"🧮 动态策略: 总步数={steps}, 验证间隔={val_int}, GPU数量={len(cfg.devices)}")

    # 【修改点 2】配置 Logger
    loggers = [
        # WandB: save_dir 设为 experiment_dir，这样 wandb 文件夹就会生成在实验目录下
        WandbLogger(
            save_dir=experiment_dir,
            name=run_name,
            config=OmegaConf.to_container(cfg),
            offline=True,
            mode="offline"
        ),
        # CSV: 使用自定义 CleanCSVLogger
        # save_dir 设为 experiment_dir，同时把 name 和 version 设为空
        # 这样 metrics.csv 就会直接生成在 experiment_dir 下，且没有 hparams.yaml
        CleanCSVLogger(
            save_dir=experiment_dir,
            name="",
            version=""
        )
    ]

    # ModelCheckpoint
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{cfg.chkpt_folder}/{cfg.data_name}/{os.path.basename(os.path.normpath(cfg.data[d_name].path))}/{run_name}",
        monitor="val_DiceMetric",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="{step}-dice:{val_DiceMetric:.2f}-" + f"{len(cfg.devices)}GPU",
        auto_insert_metric_name=False
    )
    ckpt_cb.CHECKPOINT_EQUALS_CHAR = ":"
    ckpt_cb.CHECKPOINT_NAME_LAST = run_name + "_last"

    train_dl = get_loader(cfg, "train", rank)
    val_dl = get_loader(cfg, "val", rank)
    test_dl = get_loader(cfg, "test", rank)

    model = hydra.utils.instantiate(cfg.model)
    safe_load_weights(model, cfg.path_to_chkpt, rank)

    pl_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(model=model, evaluator=Evaluator(),
                                                                      dataset_name=d_name)

    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer,
                                      logger=loggers, callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
                                      max_steps=steps, val_check_interval=val_int, devices=cfg.devices,
                                      accelerator="gpu", strategy="ddp", sync_batchnorm=True,
                                      use_distributed_sampler=False,
                                      enable_model_summary=False
                                      )()

    if cfg.num_shots == 0:
        if rank == 0: logger.info("🚀 开始零样本评估 (Zero-Shot)")
        trainer.test(pl_module, test_dl)
    else:
        if rank == 0: logger.info("🚀 开始微调流程 (初始验证 -> 训练 -> 测试)")
        trainer.validate(pl_module, val_dl)
        trainer.fit(pl_module, train_dl, val_dl)
        trainer.test(pl_module, test_dl, ckpt_path="best")

    if rank == 0:
        res = trainer.callback_metrics
        logger.info(f"\n🏆 最终测试 Dice: {res.get('test_DiceMetric', 0):.4f} | 日志路径: {experiment_dir}")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()