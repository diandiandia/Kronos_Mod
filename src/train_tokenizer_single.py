import os
import json
import time
from time import gmtime, strftime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
import psutil
import numpy as np

from finetune.config import Config as TrainingConfig
from finetune.dataset import QlibDataset
from models.kronos import KronosTokenizer
from models.enhanced_kronos import create_enhanced_tokenizer
from utils.training_utils import set_seed, get_model_size, format_time

# 设置日志
import logging
from utils.training_utils import get_device_name

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# 内存监控工具
class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage(self):
        return self.process.memory_info().rss / 1024 / 1024  # MB
    
    def get_gpu_memory(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        return 0
    
    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_dataloaders(config):
    """创建单机版数据加载器 - 优化版本"""
    print("创建数据加载器...")
    
    # 内存监控
    memory_monitor = MemoryMonitor()
    initial_memory = memory_monitor.get_memory_usage()
    
    # 预加载和缓存数据集信息
    try:
        train_dataset = QlibDataset("train")
        valid_dataset = QlibDataset("val")
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        raise

    print(f"训练数据集大小: {len(train_dataset)}, 验证数据集大小: {len(valid_dataset)}")
    
    # 优化数据加载器配置
    # 根据可用内存动态调整num_workers
    available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
    optimal_workers = min(config.num_workers, max(1, int(available_memory / 1024)))
    
    # 单机版使用RandomSampler
    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=optimal_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        persistent_workers=True if optimal_workers > 0 else False,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        persistent_workers=True if optimal_workers > 0 else False,
    )
    
    final_memory = memory_monitor.get_memory_usage()
    print(f"数据加载器创建完成。内存使用: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
    print(f"训练步数/epoch: {len(train_loader)}, 验证步数: {len(val_loader)}, 工作进程: {optimal_workers}")
    
    return train_loader, val_loader, train_dataset, valid_dataset


def compute_adaptive_loss_weights(recon_loss, bsq_loss, epoch, total_epochs):
    """自适应损失权重计算"""
    # 随着训练进行，逐渐增加重建损失的权重
    progress = epoch / total_epochs
    recon_weight = 0.5 + 0.3 * progress  # 从0.5增加到0.8
    bsq_weight = 1.0 - recon_weight
    return recon_weight, bsq_weight

def train_model(model, device, config, save_dir):
    """单机版训练循环 - 优化版本"""
    start_time = time.time()
    memory_monitor = MemoryMonitor()
    
    effective_bs = config.batch_size * config.accumulation_steps
    print(f"批次大小: {config.batch_size}")
    print(f"有效总批次大小: {effective_bs}")
    print(f"初始内存使用: {memory_monitor.get_memory_usage():.1f}MB")
    print(f"初始GPU内存: {memory_monitor.get_gpu_memory():.1f}MB")

    try:
        train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config)
    except Exception as e:
        logger.error(f"数据加载器创建失败: {e}")
        raise

    # 优化器配置
    optimizer = AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 双重学习率调度策略
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(0.1 * total_steps)

    # 主调度器
    main_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        ],
        milestones=[warmup_steps]
    )
    
    # 备用调度器用于验证损失不改善时
    backup_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    dt_result = {}
    batch_idx_global_train = 0
    patience_counter = 0
    loss_history = []
    val_loss_history = []

    for epoch_idx in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss_sum = 0.0
        num_batches = 0

        # 设置数据集随机种子
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        valid_dataset.set_epoch_seed(0)

        for i, (ori_batch_x, _) in enumerate(train_loader):
            try:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)

                # --- 优化的梯度累积循环 ---
                current_batch_total_loss = 0.0
                for j in range(config.accumulation_steps):
                    start_idx = j * (ori_batch_x.shape[0] // config.accumulation_steps)
                    end_idx = (j + 1) * (ori_batch_x.shape[0] // config.accumulation_steps)
                    batch_x = ori_batch_x[start_idx:end_idx]

                    # 前向传播
                    zs, bsq_loss, _, _ = model(batch_x)
                    z_pre, z = zs

                    # 自适应损失权重
                    recon_weight, bsq_weight = compute_adaptive_loss_weights(
                        F.mse_loss(z_pre, batch_x).item(), 
                        bsq_loss.item(), 
                        epoch_idx, 
                        config.epochs
                    )
                    
                    # 损失计算
                    recon_loss_pre = F.mse_loss(z_pre, batch_x)
                    recon_loss_all = F.mse_loss(z, batch_x)
                    recon_loss = recon_loss_pre + recon_loss_all
                    loss = recon_weight * recon_loss + bsq_weight * bsq_loss

                    loss_scaled = loss / config.accumulation_steps
                    current_batch_total_loss += loss.item()
                    loss_scaled.backward()

                # --- 优化的梯度累积后处理 ---
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=2.0)

                # 增强的梯度监控
                if batch_idx_global_train % 10 == 0:
                    total_norm = 0
                    grad_stats = []
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            grad_stats.append(f"{name}: {param_norm.item():.6f}")
                    total_norm = total_norm ** 0.5
                    
                    if batch_idx_global_train % 50 == 0:  # 每50步详细记录
                        logger.info(f"Tokenizer梯度范数: {total_norm:.6f}")
                        logger.debug(f"各层梯度: {', '.join(grad_stats[:3])}...")

                optimizer.step()
                main_scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss_sum += current_batch_total_loss / config.accumulation_steps
                num_batches += 1

                # --- 增强的日志记录 ---
                if (batch_idx_global_train + 1) % config.log_interval == 0:
                    avg_loss = current_batch_total_loss / config.accumulation_steps
                    current_lr = optimizer.param_groups[0]['lr']
                    memory_usage = memory_monitor.get_memory_usage()
                    gpu_memory = memory_monitor.get_gpu_memory()
                    
                    print(
                        f"[Epoch {epoch_idx + 1}/{config.epochs}, Step {i + 1}/{len(train_loader)}] "
                        f"LR {current_lr:.6f}, Loss: {avg_loss:.4f}, "
                        f"Mem: {memory_usage:.1f}MB, GPU: {gpu_memory:.1f}MB"
                    )

                batch_idx_global_train += 1
                
                # 定期内存清理
                if batch_idx_global_train % 100 == 0:
                    memory_monitor.cleanup()
                    
            except Exception as e:
                logger.error(f"训练步骤 {i} 出错: {e}")
                continue

        # --- 增强的验证循环 ---
        model.eval()
        tot_val_loss_sum = 0.0
        val_sample_count = 0
        val_recon_losses = []
        val_bsq_losses = []

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                try:
                    ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                    zs, bsq_loss, _, _ = model(ori_batch_x)
                    _, z = zs
                    
                    val_loss_item = F.mse_loss(z, ori_batch_x)
                    val_recon_losses.append(val_loss_item.item())
                    val_bsq_losses.append(bsq_loss.item())

                    tot_val_loss_sum += val_loss_item.item() * ori_batch_x.size(0)
                    val_sample_count += ori_batch_x.size(0)
                except Exception as e:
                    logger.error(f"验证步骤出错: {e}")
                    continue

        avg_val_loss = tot_val_loss_sum / val_sample_count if val_sample_count > 0 else 0
        avg_epoch_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
        
        # 记录损失历史
        loss_history.append(avg_epoch_loss)
        val_loss_history.append(avg_val_loss)
        
        # 计算验证集上的详细指标
        val_recon_mean = np.mean(val_recon_losses) if val_recon_losses else 0
        val_bsq_mean = np.mean(val_bsq_losses) if val_bsq_losses else 0

        # --- 增强的周期结束总结 ---
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        current_memory = memory_monitor.get_memory_usage()
        current_gpu_memory = memory_monitor.get_gpu_memory()
        
        print(f"\n--- Epoch {epoch_idx + 1}/{config.epochs} 总结 ---")
        print(f"训练损失: {avg_epoch_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        print(f"验证重建损失: {val_recon_mean:.4f}, 验证BSQ损失: {val_bsq_mean:.4f}")
        print(f"本周期时间: {format_time(epoch_time)}, 总耗时: {format_time(total_time)}")
        print(f"内存使用: {current_memory:.1f}MB, GPU内存: {current_gpu_memory:.1f}MB\n")

        # 优化的早停策略
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model_epoch_{epoch_idx + 1}"
            model.save_pretrained(save_path)
            print(f"最佳模型保存到 {save_path} (验证损失: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"验证损失未改善: {patience_counter} 次 (最佳: {best_val_loss:.4f})")

            # 使用备用调度器
            backup_scheduler.step(avg_val_loss)

        # 早停机制 - 基于趋势判断
        if patience_counter >= 8:
            # 检查最近几个epoch的损失趋势
            if len(val_loss_history) >= 5:
                recent_trend = np.mean(val_loss_history[-5:]) - np.mean(val_loss_history[-10:-5])
                if recent_trend > 0:  # 损失在上升
                    print(f"早停触发，在第 {epoch_idx + 1} 个epoch停止训练 (损失上升趋势)")
                    break
                else:
                    print(f"损失趋于平稳，继续训练...")
            else:
                print(f"早停触发，在第 {epoch_idx + 1} 个epoch停止训练")
                break

    # 训练完成后的总结
    dt_result["best_val_loss"] = best_val_loss
    dt_result["loss_history"] = loss_history
    dt_result["val_loss_history"] = val_loss_history
    dt_result["total_epochs"] = epoch_idx + 1
    dt_result["total_training_time"] = format_time(time.time() - start_time)
    
    return model, dt_result


def setup_experiment_environment(config):
    """设置实验环境"""
    # 设置随机种子
    set_seed(config.seed)
    
    # 选择设备
    device = get_device_name()
    print(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    return device, save_dir, log_dir

def save_training_summary(save_dir, master_summary, dt_result):
    """保存训练总结"""
    master_summary["final_result"] = dt_result
    summary_path = os.path.join(save_dir, "summary.json")
    
    try:
        with open(summary_path, "w") as f:
            json.dump(master_summary, f, indent=4)
        print(f"训练总结已保存到: {summary_path}")
    except Exception as e:
        logger.error(f"保存训练总结失败: {e}")

def main():
    """单机版主函数 - 优化版本"""
    try:
        # 加载配置
        config = TrainingConfig()
        
        # 设置实验环境
        device, save_dir, log_dir = setup_experiment_environment(config)
        
        # 记录开始时间和系统信息
        import platform
        master_summary = {
            "start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            "save_directory": save_dir,
            "device": str(device),
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            }
        }
        
        # 初始化增强模型
        print("初始化模型...")
        try:
            model = create_enhanced_tokenizer(
                config.pretrained_tokenizer_path,
                dropout_rate=config.dropout_rate,
                use_batch_norm=config.use_batch_norm
            )
            model.to(device)
            print(f"使用增强Tokenizer: Dropout={config.dropout_rate}, BatchNorm={config.use_batch_norm}")
            print(f"模型参数量: {get_model_size(model)}")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
        
        # 开始训练
        print("开始训练...")
        try:
            _, dt_result = train_model(model, device, config, save_dir)
            print("训练完成。")
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
        finally:
            # 最终内存清理
            memory_monitor = MemoryMonitor()
            memory_monitor.cleanup()
        
        # 保存总结
        save_training_summary(save_dir, master_summary, dt_result)
        
        # 打印最终结果
        if "best_val_loss" in dt_result:
            print(f"\n=== 训练完成 ===")
            print(f"最佳验证损失: {dt_result['best_val_loss']:.4f}")
            if "total_epochs" in dt_result:
                print(f"训练轮数: {dt_result['total_epochs']}")
            if "total_training_time" in dt_result:
                print(f"总训练时间: {dt_result['total_training_time']}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断。")
        return
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    # 单机版运行：python train_tokenizer_single.py
    main()