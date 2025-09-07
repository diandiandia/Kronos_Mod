import os
import sys
import json
import time
from time import gmtime, strftime
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from finetune.config import Config as TrainingConfig
from finetune.dataset import QlibDataset
from models.kronos import KronosTokenizer
from utils.training_utils import set_seed, get_model_size, format_time

# 设置日志
import logging
from utils.training_utils import get_device_name

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def create_dataloaders(config):
    """创建单机版数据加载器"""
    print("创建数据加载器...")

    train_dataset = QlibDataset("train")
    valid_dataset = QlibDataset("val")

    print(f"训练数据集大小: {len(train_dataset)}, 验证数据集大小: {len(valid_dataset)}")

    # 单机版使用RandomSampler
    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    print(
        f"数据加载器创建完成。训练步数/epoch: {len(train_loader)}, 验证步数: {len(val_loader)}"
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, device, config, save_dir):
    """单机版训练循环"""
    start_time = time.time()

    effective_bs = config.batch_size * config.accumulation_steps
    print(f"批次大小: {config.batch_size}")
    print(f"有效总批次大小: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay,
    )

    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        pct_start=0.03,
        div_factor=10,
    )

    best_val_loss = float("inf")
    dt_result = {}
    batch_idx_global_train = 0

    for epoch_idx in range(config.epochs):
        epoch_start_time = time.time()
        model.train()

        # 设置数据集随机种子
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        valid_dataset.set_epoch_seed(0)  # 保持验证采样一致

        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)

            # --- 梯度累积循环 ---
            current_batch_total_loss = 0.0
            for j in range(config.accumulation_steps):
                start_idx = j * (ori_batch_x.shape[0] // config.accumulation_steps)
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config.accumulation_steps)
                batch_x = ori_batch_x[start_idx:end_idx]

                # 前向传播
                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs

                # 损失计算
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2  # 假设 w_1=w_2=1

                loss_scaled = loss / config.accumulation_steps
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            # --- 累积后的优化器步骤 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # --- 日志记录 ---
            if (batch_idx_global_train + 1) % config.log_interval == 0:
                avg_loss = current_batch_total_loss / config.accumulation_steps
                print(
                    f"[Epoch {epoch_idx + 1}/{config.epochs}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )

            batch_idx_global_train += 1

        # --- 验证循环 ---
        model.eval()
        tot_val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)

                tot_val_loss_sum += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count += ori_batch_x.size(0)

        avg_val_loss = (
            tot_val_loss_sum / val_sample_count if val_sample_count > 0 else 0
        )

        # --- 周期结束总结和检查点保存 ---
        print(f"\n--- Epoch {epoch_idx + 1}/{config.epochs} 总结 ---")
        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"本周期时间: {format_time(time.time() - epoch_start_time)}")
        print(f"总耗时: {format_time(time.time() - start_time)}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"最佳模型保存到 {save_path} (验证损失: {best_val_loss:.4f})")

    dt_result["best_val_loss"] = best_val_loss
    return model, dt_result


def main():
    """单机版主函数"""
    # 加载配置
    config = TrainingConfig()

    # 设置随机种子
    set_seed(config.seed)

    # 选择设备
    device = get_device_name()
    print(f"使用设备: {device}")

    # 创建保存目录
    save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    # 记录开始时间
    master_summary = {
        "start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        "save_directory": save_dir,
        "device": str(device),
    }

    # 初始化模型
    model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    model.to(device)

    print(f"模型参数量: {get_model_size(model)}")

    # 开始训练
    print("开始训练...")
    _, dt_result = train_model(model, device, config, save_dir)

    # 保存总结
    master_summary["final_result"] = dt_result
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(master_summary, f, indent=4)

    print("训练完成。总结文件已保存。")


if __name__ == "__main__":
    # 单机版运行：python train_tokenizer_single.py
    main()