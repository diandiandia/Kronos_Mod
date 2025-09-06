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
        self.dataset_end_time = '2025-09-04'

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
        self.train_time_range = ["2005-01-04", "2024-06-30"]  # 训练集：留出90天缓冲区
        self.val_time_range = ["2024-10-01", "2025-01-31"]    # 验证集：从训练集结束+90天后开始
        self.test_time_range = ["2025-02-01", "2025-09-04"]   # 测试集：从验证集结束+90天后开始
        self.backtest_time_range = ["2025-02-01", "2025-09-05"] # 回测：与测试集保持一致

        # TODO: Directory to save the processed, pickled datasets.
        self.dataset_path = "data/processed_datasets"

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.num_workers = 4

        self.epochs = 50  # 增加epoch数，配合早停机制
        self.log_interval = 100  # Log training status every N batches.
        self.batch_size = 32  # 减小batch size以增强正则化效果

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        self.n_train_iter = 1500 * self.batch_size  # 减少训练样本数
        self.n_val_iter = 300 * self.batch_size     # 减少验证样本数

        # Learning rates for different model components。
        self.tokenizer_learning_rate = 1e-4  # 降低学习率
        self.predictor_learning_rate = 2e-5  # 降低学习率

        # Gradient accumulation to simulate a larger batch size。
        self.accumulation_steps = 1

        # AdamW optimizer parameters。
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.2  # 增加L2正则化

        # Miscellaneous
        self.seed = 100  # Global random seed for reproducibility.

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = True # Set to False if you don't want to use Comet ML
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace" # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = 'finetune_demo'
        self.comet_name = 'finetune_demo'

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = 'finetune_tokenizer'
        self.predictor_save_folder_name = 'finetune_predictor'
        self.backtest_save_folder_name = 'finetune_backtest'

        # Path for backtesting results.
        self.backtest_result_path = "./outputs/backtest_results"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # TODO: Update these paths to your pretrained model locations.
        # These can be local paths or Hugging Face Hub model identifiers.
        self.pretrained_tokenizer_path = "pretrained/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "pretrained/Kronos-small"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 5  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
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