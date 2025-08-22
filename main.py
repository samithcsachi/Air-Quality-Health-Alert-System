from Air_Quality_Health_Alert_System import logger

from Air_Quality_Health_Alert_System.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Air_Quality_Health_Alert_System.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from Air_Quality_Health_Alert_System.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from Air_Quality_Health_Alert_System.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from Air_Quality_Health_Alert_System.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from Air_Quality_Health_Alert_System.pipeline.stage_06_alert_generation import AlertGenerationTrainingPipeline


STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"


try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"


try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer Stage"


try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation stage"


try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    model_evaluator = ModelEvaluationTrainingPipeline()
    model_evaluator.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Alert Generation stage"



try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    alert_generator = AlertGenerationTrainingPipeline()
    alert_generator.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
