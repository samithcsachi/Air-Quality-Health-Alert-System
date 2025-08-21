import os
import urllib.request as request
from Air_Quality_Health_Alert_System import logger
from Air_Quality_Health_Alert_System.utils.common import get_size
from pathlib import Path
from Air_Quality_Health_Alert_System.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
    
        local_path = Path(self.config.local_data_file)

      
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():
            try:
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=str(local_path)
                )
                logger.info(f"{filename} downloaded with headers: \n{headers}")
                return filename
            except Exception as e:
                logger.error(f"Failed to download file: {e}")
                raise
        else:
            logger.info(f"File already exists: {local_path}")
            return str(local_path)