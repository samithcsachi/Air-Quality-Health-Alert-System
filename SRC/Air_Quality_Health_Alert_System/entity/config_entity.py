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
