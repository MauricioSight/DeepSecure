import mlflow
from loggers.base_tracker import BaseTracker

class MLflowTracker(BaseTracker):
    def start_run(self, run_name: str, config: dict):
        mlflow.start_run(run_name=run_name)
        for k, v in config.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict, step: int):
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_artifact(self, filepath: str):
        mlflow.log_artifact(filepath)

    def finish(self):
        mlflow.end_run()
