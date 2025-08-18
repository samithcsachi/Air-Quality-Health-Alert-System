import os
from Air_Quality_Health_Alert_System import logger
from Air_Quality_Health_Alert_System.entity.config_entity  import DataValidationConfig
from pathlib import Path
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self) -> bool:
        try:
            df = pd.read_csv(self.config.data_dir)
            all_cols = list(df.columns)
            all_schema = self.config.all_schema.keys()
            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"validation status: {validation_status}")
            return validation_status
        except Exception as e:
            raise e