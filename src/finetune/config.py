import os

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # TODO: Update this path to your Qlib data directory.
        self.qlib_data_path = "qlib_data/cn_data"
        self.instrument = 'csi300'

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2005-01-04"
        self.dataset_end_time = '2025-09-05'

        # Sliding window parameters for creating samples.
        self.lookback_window = 90  # Number of past time steps for input.
        self.predict_window = 10  # Number of future time steps for prediction.
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Note: Time ranges are carefully adjusted to ensure no data leakage
        # considering the 90-day lookback_window for feature generation.
        self.train_time_range = ["2005-01-04", "2024-05-31"]  # 训练集：留出足够缓冲区
        self.val_time_range = ["2024-06-01", "2024-12-31"]    # 验证集：确保有足够数据
        self.test_time_range = ["2025-01-01", "2025-09-05"]   # 测试集：最新数据
        self.backtest_time_range = ["2025-01-01", "2025-09-05"] # 回测：与测试集保持一致

        # TODO: Directory to save the processed, pickled datasets.
        self.dataset_path = "data/processed_datasets"

        # =================================================================
        # Training Hyperparameters - 当前训练状态调整
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.num_workers = 4

        self.epochs = 30  # 保持最大epoch数
        self.log_interval = 50  # 日志记录频率
        self.batch_size = 16  # 小批次增强正则化
        
        # 基于当前训练状态调整早停参数
        self.early_stopping_patience = 7  # 增加耐心值，给模型更多机会
        self.min_delta = 5e-4  # 略微提高改善阈值

        # Number of samples to draw for one "epoch" of training/validation.
        self.n_train_iter = 1000 * self.batch_size
        self.n_val_iter = 200 * self.batch_size

        # Learning rates - 当前已处于较低水平
        self.tokenizer_learning_rate = 5e-5
        self.predictor_learning_rate = 1e-5

        # Gradient accumulation
        self.accumulation_steps = 1

        # AdamW optimizer参数
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.3  # 保持强正则化

        # 正则化参数
        self.dropout_rate = 0.2
        self.label_smoothing = 0.1

        # 随机种子
        self.seed = 42

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = True
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace"
        }
        self.comet_tag = 'finetune_demo_optimized_v2'
        self.comet_name = 'kronos_training_final'

        # 保存路径
        self.save_path = "./outputs/models_optimized"
        self.tokenizer_save_folder_name = 'finetune_tokenizer'
        self.predictor_save_folder_name = 'finetune_predictor'
        self.backtest_save_folder_name = 'finetune_backtest'
        self.backtest_result_path = "./outputs/backtest_results_optimized"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        self.pretrained_tokenizer_path = "pretrained/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "pretrained/Kronos-base"
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")