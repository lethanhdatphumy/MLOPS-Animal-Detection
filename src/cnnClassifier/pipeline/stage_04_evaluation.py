# Stage 4: evaluates the trained model and writes loss/accuracy to scores.json
from src.cnnClassifier import logger
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.evaluation import Evaluator
from pathlib import Path

STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:
    def __init__(self):
        """Initialize the configuration manager."""
        self.config = ConfigurationManager()

    def main(self):
        """Run evaluation and write scores to scores.json."""
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        eval_config = self.config.get_evaluation_config()
        evaluator = Evaluator(config=eval_config)
        loss, acc = evaluator.evaluate()
        scores_path = Path.cwd() / "scores.json"
        evaluator.write_scores(loss, acc, scores_path)
        logger.info(f"Saved scores to {scores_path}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


if __name__ == "__main__":
    try:
        pipe = EvaluationPipeline()
        pipe.main()
    except Exception as e:
        logger.exception(e)
        raise e
