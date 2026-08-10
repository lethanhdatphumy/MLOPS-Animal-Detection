# Stage 2: downloads pretrained VGG16 and attaches the custom classification head
from src.cnnClassifier import logger
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.prepare_base_model import PrepareBaseModel

STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """Save the pretrained base model and then the version with the custom head."""
        config = ConfigurationManager()
        prepare_cfg = config.get_prepare_base_model_config()
        preparer = PrepareBaseModel(config=prepare_cfg)
        preparer.get_base_model()
        preparer.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
