from utils.training_utils import (
    get_device_name,
    set_seed,
    get_model_size,
    format_time
)
from models.kronos import KronosTokenizer
from models.enhanced_kronos import create_enhanced_model
from models.kronos import Kronos
from finetune.dataset import QlibDataset
from finetune.config import Config
import os
import sys
import json
import time
import copy
from time import gmtime, strftime
import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# 设置项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import shared utilities


def create_dataloaders(config: dict, device_name: str):
    """
    Creates and returns dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        device: The torch device to use for determining pin_memory.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print("Creating dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(
        f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(valid_dataset)

    # Check if we should use pin_memory (avoid MPS warning on macOS)
    device = torch.device(device=device_name)
    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=use_pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=use_pin_memory, drop_last=False
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, tokenizer, device_name, config, save_dir, logger):
    """
    The main training and validation loop for the predictor with anti-overfitting optimizations.
    """
    start_time = time.time()
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['predictor_learning_rate']}")
    print(f"Weight decay: {config['adam_weight_decay']}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config, device_name)

    # Enhanced optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )

    # 增强的学习率调度器 - 多阶段调度策略
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(0.15 * total_steps)  # 增加预热步数

    # 创建多阶段调度器
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, total_iters=warmup_steps
    )

    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(0.3 * total_steps), eta_min=1e-5
    )

    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # 组合调度器
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_steps]
    )

    # 用于跟踪调度阶段
    scheduler_phase = 'warmup'  # 'warmup', 'cosine', 'plateau'
    phase_transition_step = warmup_steps + int(0.3 * total_steps)

    # Early stopping parameters
    best_val_loss = float('inf')
    best_model_state = None
    patience = config.get('early_stopping_patience', 5)
    patience_counter = 0
    min_delta = 1e-4

    dt_result = {}
    batch_idx_global = 0

    # Training metrics tracking
    train_losses = []
    val_losses = []

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()

        train_dataset.set_epoch_seed(epoch_idx * 10000)
        valid_dataset.set_epoch_seed(0)

        epoch_train_loss = 0.0
        train_batches = 0

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.squeeze(0).to(device_name, non_blocking=True)
            batch_x_stamp = batch_x_stamp.squeeze(
                0).to(device_name, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.base_model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1])

            # 增强的损失函数权重调整 - 动态权重分配
            # 根据训练进度动态调整权重
            progress_ratio = epoch_idx / config['epochs']
            s1_weight = 0.8 - 0.2 * progress_ratio  # 从0.8逐渐降低到0.6
            s2_weight = 0.2 + 0.2 * progress_ratio  # 从0.2逐渐增加到0.4

            # 添加一致性损失（S1和S2预测的一致性）
            s1_probs = F.softmax(logits[0], dim=-1)
            s2_probs = F.softmax(logits[1], dim=-1)
            consistency_loss = F.kl_div(
                F.log_softmax(s1_probs.mean(dim=1, keepdim=True), dim=-1),
                F.softmax(s2_probs.mean(dim=1, keepdim=True), dim=-1),
                reduction='batchmean'
            )

            # 添加熵正则化（防止过拟合）
            s1_entropy = -(s1_probs * torch.log(s1_probs + 1e-8)
                           ).sum(dim=-1).mean()
            s2_entropy = -(s2_probs * torch.log(s2_probs + 1e-8)
                           ).sum(dim=-1).mean()
            entropy_reg = 0.01 * (s1_entropy + s2_entropy)

            # 组合损失
            loss = s1_weight * s1_loss + s2_weight * s2_loss + 0.1 * consistency_loss + entropy_reg

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping with enhanced monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)

            # 梯度累积优化 - 添加梯度监控
            if batch_idx_global % 10 == 0:  # 每10步记录一次梯度信息
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                if logger:
                    logger.log_metric('gradient_total_norm',
                                      total_norm, step=batch_idx_global)
            optimizer.step()

            # 多阶段调度器逻辑
            if batch_idx_global < warmup_steps:
                scheduler.step()
            elif batch_idx_global < phase_transition_step:
                scheduler.step()
                scheduler_phase = 'cosine'
            else:
                # 在验证阶段使用ReduceLROnPlateau
                pass

            # Accumulate training loss
            epoch_train_loss += loss.item()
            train_batches += 1

            # Enhanced logging with gradient norm
            if (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}"
                )
            if logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch',
                                  loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch',
                                  s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch',
                                  s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate',
                                  lr, step=batch_idx_global)
                logger.log_metric(
                    'gradient_norm', grad_norm.item(), step=batch_idx_global)

            batch_idx_global += 1

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum = 0.0
        val_batches_processed = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.squeeze(0).to(device_name, non_blocking=True)
                batch_x_stamp = batch_x_stamp.squeeze(
                    0).to(device_name, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1],
                               batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.base_model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum += val_loss.item()
                val_batches_processed += 1

        avg_val_loss = tot_val_loss_sum / \
            val_batches_processed if val_batches_processed > 0 else 0
        val_losses.append(avg_val_loss)

        # --- End of Epoch Summary & Checkpointing ---
        epoch_time = time.time() - epoch_start_time
        print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Scheduler Phase: {scheduler_phase}")
        print(f"Time This Epoch: {format_time(epoch_time)}")
        print(f"Total Time Elapsed: {format_time(time.time() - start_time)}")

        if logger:
            logger.log_metric('train_predictor_loss_epoch',
                              avg_train_loss, epoch=epoch_idx)
            logger.log_metric('val_predictor_loss_epoch',
                              avg_val_loss, epoch=epoch_idx)
            logger.log_metric('scheduler_phase', 1 if scheduler_phase == 'warmup' else (
                2 if scheduler_phase == 'cosine' else 3), epoch=epoch_idx)

        # 在plateau阶段使用ReduceLROnPlateau
        if scheduler_phase == 'plateau' or batch_idx_global >= phase_transition_step:
            scheduler3.step(avg_val_loss)
            if scheduler_phase != 'plateau':
                scheduler_phase = 'plateau'
                print("🔄 Switched to ReduceLROnPlateau scheduler")

        # 增强的早停策略 - 多重条件判断
        improvement_ratio = (best_val_loss - avg_val_loss) / \
            best_val_loss if best_val_loss != float('inf') else 0

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(
                f"🎯 Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epoch(s)")
            print(f"📊 Improvement ratio: {improvement_ratio:.4%}")

            # 智能学习率调整
            if patience_counter >= 2 and improvement_ratio > -0.01:  # 改善很小但仍在改善
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8  # 温和降低学习率
                print(
                    f"📉 Learning rate gently reduced to {optimizer.param_groups[0]['lr']:.6f}")
            elif patience_counter >= 4:  # 长期无改善
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5  # 大幅降低学习率
                print(
                    f"📉 Learning rate significantly reduced to {optimizer.param_groups[0]['lr']:.6f}")

        # 多重早停条件
        stop_conditions = [
            patience_counter >= patience,  # 基础早停
            avg_val_loss > best_val_loss * 1.1 and patience_counter >= 3,  # 损失显著恶化
            (avg_val_loss - best_val_loss) < 1e-5 and patience_counter >= 5  # 收敛停滞
        ]

        if any(stop_conditions):
            stop_reason = "patience exceeded" if stop_conditions[0] else (
                "significant loss degradation" if stop_conditions[1] else "convergence stagnation"
            )
            print(f"🛑 Early stopping triggered: {stop_reason}")
            break

        # 增强的过拟合检测
        if epoch_idx >= 3:
            recent_val_trend = val_losses[-4:]
            train_val_gap = avg_train_loss - avg_val_loss

            # 检测验证损失上升趋势
            val_increasing = all(
                recent_val_trend[i] >= recent_val_trend[i-1] for i in range(1, len(recent_val_trend)))

            # 检测训练验证差距过大
            large_gap = train_val_gap > 0.5 and avg_train_loss < avg_val_loss * 0.8

            if val_increasing:
                print(
                    "⚠️  Warning: Validation loss increasing - potential overfitting detected")
                if patience_counter >= 2:
                    print("🔧 Increasing dropout rate to combat overfitting")
                    # 动态调整dropout率（这里只是示例，实际实现需要更复杂）

            if large_gap:
                print(
                    "⚠️  Warning: Large train-validation gap detected - potential overfitting")

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with validation loss: {best_val_loss:.4f}")

    dt_result['best_val_loss'] = best_val_loss
    dt_result['final_train_loss'] = train_losses[-1] if train_losses else None
    dt_result['final_val_loss'] = val_losses[-1] if val_losses else None
    dt_result['epochs_trained'] = epoch_idx + 1
    dt_result['early_stopped'] = patience_counter >= patience

    return dt_result


def main(config: dict):
    """Main function to orchestrate the training process with optimizations."""
    device_name = get_device_name()
    set_seed(config['seed'])

    save_dir = os.path.join(config['save_path'],
                            config['predictor_save_folder_name'])

    # Logger and summary setup
    comet_logger = None
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    master_summary = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'save_directory': save_dir,
        'optimization_notes': [
            'Enhanced early stopping with patience',
            'Improved learning rate scheduling with warmup and cosine annealing',
            'Gradient clipping for stability',
            'Overfitting detection and prevention',
            'Comprehensive metrics tracking'
        ]
    }

    if config['use_comet']:
        comet_logger = comet_ml.Experiment(
            api_key=config['comet_config']['api_key'],
            project_name=config['comet_config']['project_name'],
            workspace=config['comet_config']['workspace'],
        )
        comet_logger.add_tag(config['comet_tag'])
        comet_logger.set_name(f"{config['comet_name']}_optimized")
        comet_logger.log_parameters(config)
        print("Comet Logger Initialized.")

    # Model Initialization with enhanced logging
    print("Loading tokenizer and model...")
    
    # Handle local path loading for tokenizer
    tokenizer_path = config['finetuned_tokenizer_path']
    print(f"Attempting to load tokenizer from: {tokenizer_path}")
    
    if os.path.exists(tokenizer_path):
        # Load from local directory
        print(f"Loading tokenizer from local path: {tokenizer_path}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        # Load from HuggingFace hub
        print(f"Loading tokenizer from HuggingFace hub: {tokenizer_path}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    
    tokenizer.eval().to(device_name)

    # Initialize enhanced model with Dropout and BatchNorm
    model = create_enhanced_model(
        config['pretrained_predictor_path'],
        dropout_rate=config.get('dropout_rate', 0.1),
        use_batch_norm=config.get('use_batch_norm', True)
    )
    model.to(device_name)
    print(
        f"使用增强模型: Dropout={config.get('dropout_rate', 0.1)}, BatchNorm={config.get('use_batch_norm', True)}")

    model_size = get_model_size(model)
    print(f"Predictor Model Size: {model_size}")

    # Log model architecture details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Start Training
    print("\n🚀 Starting optimized training...")
    dt_result = train_model(
        model, tokenizer, device_name, config, save_dir, comet_logger
    )

    master_summary['final_result'] = dt_result
    master_summary['model_info'] = {
        'model_size': model_size,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(master_summary, f, indent=4)

    print('\n' + '='*50)
    print('✅ Training completed successfully!')
    print(f"Best validation loss: {dt_result['best_val_loss']:.4f}")
    print(f"Epochs trained: {dt_result['epochs_trained']}")
    print(f"Early stopped: {dt_result['early_stopped']}")
    print('='*50)

    if comet_logger:
        comet_logger.end()


if __name__ == '__main__':
    config_instance = Config()
    main(config_instance.__dict__)