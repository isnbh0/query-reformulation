from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = 't5'
    model_type: str = 'base'
    
    
@dataclass
class DataConfig:
    data_dir: str = 'data'
    train_file: str = 'train.json'
    valid_file: str = 'valid.json'


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 10
    save_total_limit: int = 3
    evaluation_strategy: str = 'steps'
    eval_steps: int = 10
    eval_accumulation_steps: int = 1
    eval_delay: int = 0
    eval_keep_best: bool = True
    eval_best_metric: str = 'loss'
    eval_best_mode: str = 'min'

    
@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
