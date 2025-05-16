import argparse
from datetime import datetime
import logging
import yaml

from trainers.trainer import Trainer
from models.model_factory import ModelFactory
from tracker.wandb_tracker import WandBTracker
from trainers.traning_validation import TrainValidation
from datasets.tow_ids_dataset import TOWIDSFeatureLoader
from feature_generator.sliding_window_generator import SlidingWindowGenerator
from loggers.logging import setup_logger
from utils.experiment_io import generate_run_id, get_run_dir, save_run_artifacts
from utils.seed_all import seed_all


# Shared config
config = {
        "phase": "train",
        "dataset": "TOW-IDS",
        "feature_generator": "sliding-window-generator",
        "feature_config": {
            "window_size": 44,
            "window_stride": 1,
            "number_of_bytes": 52,
        },
        "raw_x_path": "data/tow-ids-dataset/raw/Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap",
        "raw_y_path": "data/tow-ids-dataset/raw/y_train.csv",
        "processed_path": "data/tow-ids-dataset/processed",
        "processed_x_path": "data/tow-ids-dataset/processed/X_train_fg_sliding-window-generator_Wsize_44_Wstride_1_nb_52.npz",
        "processed_y_path": "data/tow-ids-dataset/processed/y_train_fg_sliding-window-generator_Wsize_44_Wstride_1_nb_52.csv",
        "model_config": {
            "model_name": "SeqWatch",
        },
        'training_config': {
            "batch_size": 32,
            "num_epochs": 15,
            "learning_rate": 0.001,
            "early_stopping_patience": 5,
            "criterion": "mean_squared_error",
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Execute train validation step')
    parser.add_argument('--model', required=False, help='YAML File containing the configs')
    args = parser.parse_args()

    if args.model is None:
        model_name = 'SeqWatch'
    else:
        model_name = args.model

    with open(f"configs/{model_name}.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_id = generate_run_id(model_name=config['model_config']['model_name'], dataset_name=config['dataset'])
    run_dir = get_run_dir(run_id)
    
    config['run_id'] = run_id

    # Setup logger
    logger = setup_logger(
        name="train_validation",
        log_file=f"{run_dir}/output.log",
        level=logging.INFO,
    )

    # log run id
    logger.info("Initiating training and validation...")
    logger.info(f"[ RUN ID: {run_id} ]")

    seed = 0
    seed_all(seed)
    logger.info(f"[ Using Seed : {seed} ]")

    # 1. Load the dataset
    logger.info("Loading data...")
    feature_generator = SlidingWindowGenerator(config, logger)
    dataset_loader = TOWIDSFeatureLoader(config, logger, feature_generator)
    values, labels = dataset_loader.load_processed(subset=config.get('dataset_subset', None))

    # 3. Initializations
    logger.info("Initializing components...")

    # 3.1 Model
    logger.info("Initializing model...")
    model_factory = ModelFactory(config)
    model = model_factory.create_model()

    # 3.2 Tracker
    logger.info("Initializing tracker...")
    tracker = WandBTracker(config={**config['training_config'], **config['model_config']}, run_name=run_id, model=model)

    # 3.3 Trainer and TrainValidation
    logger.info("Initializing trainer...")
    trainer = Trainer(config, logger, tracker, model)
    trainValidation = TrainValidation(config, logger, model, trainer, tracker)

    # 4. Execute
    logger.info("Starting execution...")
    labels, y_pred, metrics = trainValidation.execute(values, labels)
    logger.info("Execution completed.")
    
    # 5. Finish the tracker
    tracker.finish()

    # 6. Save run artifacts
    logger.info("Saving run artifacts...")
    save_run_artifacts(run_dir, model, labels, y_pred, metrics, config)
    logger.info("Run artifacts saved.")
    
    logger.info("Train and validation completed.")

if __name__ == "__main__":
    main()
