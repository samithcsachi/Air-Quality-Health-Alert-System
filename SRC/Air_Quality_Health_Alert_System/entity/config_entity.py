from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    
    root_dir: Path
    source_URL: str 
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    
    root_dir: Path
    STATUS_FILE: str
    data_dir : Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    data_transformation_dir: Path  
    model_name: str
    target_column: str
    
    # XGBoost hyperparameters
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    random_state: int

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    report_path: Path  
    target_column: str


@dataclass
class AlertConfig:
    root_dir: str
    sensitive_threshold: int
    general_threshold: int
    notification_method: str
    dashboard_refresh_interval_seconds: int = 60