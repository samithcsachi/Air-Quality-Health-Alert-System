from Air_Quality_Health_Alert_System.config.configuration import ConfigurationManager
from Air_Quality_Health_Alert_System.components.alert_generation import AlertGenerator
from Air_Quality_Health_Alert_System import logger
from pathlib import Path
import joblib

STAGE_NAME = "Alert Generation stage"


class AlertGenerationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
    
        alert_generation_config = config.get_alert_generation_config()
        alert_generator = AlertGenerator(config=alert_generation_config)

        model_artifacts = joblib.load("artifacts/model_trainer/model.joblib")
        test_data = joblib.load("artifacts/model_trainer/test_data.joblib")

        model = model_artifacts['model']
        predicted_aqi = test_data['predictions']
        actual_aqi = test_data['y_test']

        
        results = alert_generator.evaluate_alerts(predicted_aqi, actual_aqi)




if __name__ == '__main__':
    try:
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
        obj = AlertGenerationTrainingPipeline()
        obj.main()
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e