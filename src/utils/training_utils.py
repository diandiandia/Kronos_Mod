import os
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


def setup_ddp():
    """
    Initializes the distributed data parallel environment.

    This function relies on environment variables set by `torchrun` or a similar
    launcher. It initializes the process group and sets the CUDA device for the
    current process.

    Returns:
        tuple: A tuple containing (rank, world_size, local_rank).
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(
        f"[DDP Setup] Global Rank: {rank}/{world_size}, "
        f"Local Rank (GPU): {local_rank} on device {torch.cuda.current_device()}"
    )
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The base seed value.
        rank (int): The process rank, used to ensure different processes have
                    different seeds, which can be important for data loading.
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        # The two lines below can impact performance, so they are often
        # reserved for final experiments where reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """
    Calculates the number of trainable parameters in a PyTorch model and returns
    it as a human-readable string.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        str: A string representing the model size (e.g., "175.0B", "7.1M", "50.5K").
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"  # Billions
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"  # Millions
    else:
        return f"{total_params / 1e3:.1f}K"  # Thousands


def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduces a tensor's value across all processes in a distributed setup.

    Args:
        tensor (torch.Tensor): The tensor to be reduced.
        world_size (int): The total number of processes.
        op (dist.ReduceOp, optional): The reduction operation (SUM, AVG, etc.).
                                      Defaults to dist.ReduceOp.SUM.

    Returns:
        torch.Tensor: The reduced tensor, which will be identical on all processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    # Note: `dist.ReduceOp.AVG` is available in newer torch versions.
    # For compatibility, manual division is sometimes used after a SUM.
    if op == dist.ReduceOp.AVG:
        rt /= world_size
    return rt


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable H:M:S string.

    Args:
        seconds (float): The total seconds.

    Returns:
        str: The formatted time string (e.g., "0:15:32").
    """
    return str(datetime.timedelta(seconds=int(seconds)))


def get_device_name() -> str:
    """
    Returns the name of the current device (e.g., "cuda:0", "mps", "cpu").

    Returns:
        str: The name of the current device.
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    elif torch.mps.is_available():
        return "mps"
    elif torch.xpu.is_available():
        return f"xpu:{torch.xpu.current_device()}"
    else:
        return "cpu"


def create_trading_timestamps(start_date, end_date=None, start_time='09:35:00', end_time='15:00:00', 
                              lunch_start='11:30:00', lunch_end='13:00:00', freq='5min'):
    """
    创建交易时间戳序列，从start_time到end_time，排除午休时间
    
    Args:
        start_date: 开始日期，格式 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
        end_date: 结束日期，格式 'YYYY-MM-DD'，如果为None则只生成start_date一天
        start_time: 每日开始交易时间，默认 '09:35:00'
        end_time: 每日结束交易时间，默认 '15:00:00'
        lunch_start: 午休开始时间，默认 '11:30:00'
        lunch_end: 午休结束时间，默认 '13:00:00'
        freq: 时间频率，默认 '5min'
    
    Returns:
        DataFrame: 包含timestamps列的DataFrame
    """
    # 处理日期输入
    if ' ' in start_date:
        start_datetime = pd.to_datetime(start_date)
    else:
        start_datetime = pd.to_datetime(start_date + ' ' + start_time)
    
    if end_date is None:
        end_datetime = start_datetime
    else:
        end_datetime = pd.to_datetime(end_date + ' ' + end_time)
    
    # 生成完整的时间序列
    all_timestamps = pd.date_range(
        start=start_datetime,
        end=end_datetime,
        freq=freq
    )
    
    # 创建DataFrame
    df = pd.DataFrame({'timestamps': all_timestamps})
    
    # 提取时间部分
    time_part = df['timestamps'].dt.time
    
    # 转换为time对象进行比较
    start_time_obj = datetime.datetime.strptime(start_time, '%H:%M:%S').time()
    end_time_obj = datetime.datetime.strptime(end_time, '%H:%M:%S').time()
    lunch_start_obj = datetime.datetime.strptime(lunch_start, '%H:%M:%S').time()
    lunch_end_obj = datetime.datetime.strptime(lunch_end, '%H:%M:%S').time()
    
    # 过滤条件：在交易时间内且不在午休时间内
    in_trading_hours = (time_part >= start_time_obj) & (time_part <= end_time_obj)
    not_in_lunch = (time_part < lunch_start_obj) | (time_part > lunch_end_obj)
    
    # 应用过滤条件
    filtered_df = df[in_trading_hours & not_in_lunch]
    
    return filtered_df

