from Air_Quality_Health_Alert_System.config.configuration import ConfigurationManager
from Air_Quality_Health_Alert_System.components.model_evaluation import ModelEvaluation
from Air_Quality_Health_Alert_System import logger
from pathlib import Path

STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(config=model_evaluation_config)  
        model_evaluator.evaluate()




if __name__ == '__main__':
    try:
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e