from Air_Quality_Health_Alert_System.config.configuration import ConfigurationManager
from Air_Quality_Health_Alert_System.components.data_transformation import DataTransformation 
from Air_Quality_Health_Alert_System import logger

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), 'r') as file:
                status = file.read().split(" ")[-1]
                if status == "True":
                    config = ConfigurationManager()
                    data_transformation_config = config.get_data_transformation_config()
                    data_transformation = DataTransformation(config=data_transformation_config)
                    data_transformation.train_test_spliting()
                else:
                    raise Exception("Your schema is not valid")
        except Exception as e:
            print(e)
       
if __name__ == "__main__":
    try:
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e
            