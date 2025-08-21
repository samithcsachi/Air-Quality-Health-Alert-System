from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """Data Ingestion Configuration"""
    root_dir: Path
    source_URL: str 
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Data Validation Configuration"""
    root_dir: Path
    STATUS_FILE: str
    data_dir : Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    """Data Transformation Configuration"""
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Model Trainer Configuration"""
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    data_transformation_dir: Path  # ADD THIS - needed to load scaler
    model_name: str
    target_column: str
    
    # XGBoost hyperparameters
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    random_state: int
