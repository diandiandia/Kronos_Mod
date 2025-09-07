import os
import sys
import json
import time
import copy
from time import gmtime, strftime
import comet_ml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# 设置项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune.config import Config
from finetune.dataset import QlibDataset
from models.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    get_device_name,
    set_seed,
    get_model_size,
    format_time
)


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
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

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

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, device_name)

    # Enhanced optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    
    # Improved scheduler with warmup and cosine annealing
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
            )
        ],
        milestones=[warmup_steps]
    )

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
            batch_x_stamp = batch_x_stamp.squeeze(0).to(device_name, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

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
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)
                logger.log_metric('gradient_norm', grad_norm.item(), step=batch_idx_global)

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
                batch_x_stamp = batch_x_stamp.squeeze(0).to(device_name, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum += val_loss.item()
                val_batches_processed += 1

        avg_val_loss = tot_val_loss_sum / val_batches_processed if val_batches_processed > 0 else 0
        val_losses.append(avg_val_loss)

        # --- End of Epoch Summary & Checkpointing ---
        epoch_time = time.time() - epoch_start_time
        print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Time This Epoch: {format_time(epoch_time)}")
        print(f"Total Time Elapsed: {format_time(time.time() - start_time)}")
        
        if logger:
            logger.log_metric('train_predictor_loss_epoch', avg_train_loss, epoch=epoch_idx)
            logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

        # Enhanced early stopping with patience
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"🎯 Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epoch(s)")
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
            break

        # Overfitting detection
        if epoch_idx >= 2:
            recent_val_trend = val_losses[-3:]
            if all(recent_val_trend[i] >= recent_val_trend[i-1] for i in range(1, 3)):
                print("⚠️  Warning: Validation loss increasing - potential overfitting detected")

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

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

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
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device_name)

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device_name)

    model_size = get_model_size(model)
    print(f"Predictor Model Size: {model_size}")
    
    # Log model architecture details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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