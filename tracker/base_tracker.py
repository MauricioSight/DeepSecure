class BaseTracker:
    def start_run(self, run_name: str, config: dict):
        raise NotImplementedError

    def log_metrics(self, metrics: dict, step: int):
        raise NotImplementedError

    def log_artifact(self, filepath: str):
        pass  # Optional

    def finish(self):
        raise NotImplementedError
