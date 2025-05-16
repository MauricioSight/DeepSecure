import json
import yaml
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd


def generate_run_id(model_name: str, dataset_name: str, phase: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{dataset_name}_{phase}_{timestamp}"

def get_run_dir(run_id: str, base_dir: str = "runs") -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_run_artifacts(run_dir, model, labels, y_pred, metrics: dict, config: dict):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Save labels and predictions
    df = pd.DataFrame({"label": labels, "y_pred": y_pred})
    df.to_csv(run_dir / "labels_predictions.csv", index=False)

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

def save_run_tune(run_dir, tune_config: dict, train_config: dict):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "tune_config.yaml", "w") as f:
        yaml.dump(tune_config, f)

    with open(run_dir / "train_config.yaml", "w") as f:
        yaml.dump(tune_config, f)

def load_run_artifacts(run_dir):
    run_dir = Path(run_dir)

    model_state_dict = torch.load(run_dir / "model.pt", map_location='cpu')
    
    df = pd.read_csv(run_dir / "labels_predictions.csv")
    y_true = df["label"].values
    y_pred = df["y_pred"].values

    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)

    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    return {
        "model_state_dict": model_state_dict,
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "config": config
    }

def list_runs(base_dir: str = "runs"):
    base = Path(base_dir)
    return sorted([d.name for d in base.iterdir() if d.is_dir()])
