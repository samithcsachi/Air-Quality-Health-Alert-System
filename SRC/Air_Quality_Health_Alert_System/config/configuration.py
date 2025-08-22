from Air_Quality_Health_Alert_System.constants import  *
from Air_Quality_Health_Alert_System.utils.common import read_yaml, create_directories
from Air_Quality_Health_Alert_System.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig,ModelEvaluationConfig,AlertConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir=config.data_dir,
            all_schema=schema
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBOOST
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            data_transformation_dir=self.config.data_transformation.root_dir,  # ADD THIS LINE
            model_name=config.model_name,
            target_column=schema.name,
            
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            random_state=params.random_state
        )
        return model_trainer_config
    
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            report_path=config.report_path,
            target_column=schema.name
        )
        return model_evaluation_config
    

   
    def get_alert_generation_config(self) -> AlertConfig:
        config = self.config.alert_generation
        alert_config = AlertConfig(
            root_dir=config.root_dir, 
            sensitive_threshold=config.sensitive_threshold,
            general_threshold=config.general_threshold,
            notification_method=config.notification_method,
            dashboard_refresh_interval_seconds=config.dashboard.refresh_interval_seconds
        )
        return alert_config