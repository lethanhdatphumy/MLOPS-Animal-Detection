# Stage 3: fine-tunes the prepared VGG16 model on the ingested dataset
from src.cnnClassifier import logger
from src.cnnClassifier.components.training import Training
from src.cnnClassifier.components.prepare_callbacks import PrepareCallbacks
from src.cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Training Stage"


class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """Wires up callbacks and training config, then runs the training loop."""
        config = ConfigurationManager()
        training_config = config.get_training_config()
        callbacks_config = config.get_prepare_callbacks_config()

        prepare_callbacks = PrepareCallbacks(config=callbacks_config)
        tensorboard_callback = prepare_callbacks.get_tensorboard_callback()
        checkpoint_callback = prepare_callbacks.get_checkpoint_callback()

        training = Training(
            config=training_config,
            tensorboard_callback=tensorboard_callback,
            checkpoint_callback=checkpoint_callback,
        )
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e