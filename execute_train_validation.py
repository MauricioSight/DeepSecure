import logging

from trainers.trainer import Trainer
from models.model_factory import ModelFactory
from tracker.wandb_tracker import WandBTracker
from trainers.traning_validation import TrainValidation
from datasets.tow_ids_dataset import TOWIDSFeatureLoader
from feature_generator.sliding_window_generator import SlidingWindowGenerator
from loggers.logging import Logger
from utils.config_handle import load_config, flatten_dict
from utils.experiment_io import generate_run_id, get_run_dir, save_run_artifacts
from utils.seed_all import seed_all

def main(config=None, values=None, labels=None):
    config = load_config(config)

    if not hasattr(config, 'run_id'):
        run_id = generate_run_id(model_name=config['model']['name'], dataset_name=config['dataset']['name'], phase=config['phase'])
        config['run_id'] = run_id
    
    run_id = config['run_id']
    run_dir = get_run_dir(run_id)

    # Setup logger
    logger = Logger(name="train_validation", log_file=f"{run_dir}/output.log", 
                    level=logging.DEBUG if hasattr(config, 'debug') else logging.INFO)

    # log run id
    logger.info("Initiating training and validation...")
    logger.info(f"[ RUN ID: {run_id} ]")

    seed = 0
    seed_all(seed)
    logger.debug(f"[ Using Seed : {seed} ]")

    # 1. Load the dataset
    if values is None or labels is None:
        logger.debug("Loading data...")
        feature_generator = SlidingWindowGenerator(config, logger)
        dataset_loader = TOWIDSFeatureLoader(config, logger, feature_generator)
        values, labels = dataset_loader.load_processed()
        logger.info("Data loaded successfully.")
    else:
        logger.info("Using provided data for training and validation.")

    # 3. Initializations
    logger.debug("Initializing components...")

    # 3.1 Model
    logger.debug("Initializing model...")
    model_factory = ModelFactory(config)
    model = model_factory.create_model()

    # 3.2 Tracker
    logger.debug("Initializing tracker...")
    flat_config = flatten_dict(config)
    tracker = WandBTracker(config=flat_config, run_name=run_id, model=model)

    # 3.3 Trainer and TrainValidation
    logger.debug("Initializing trainer...")
    trainer = Trainer(config, logger, tracker, model)
    trainValidation = TrainValidation(config, logger, model, trainer, tracker)

    # 4. Execute
    logger.debug("Starting execution...")
    label_values, y_pred, metrics = trainValidation.execute(values, labels)
    logger.info("Execution completed.")
    
    # 5. Finish the tracker
    tracker.finish()

    # 6. Save run artifacts
    logger.debug("Saving run artifacts...")
    save_run_artifacts(run_dir, model, label_values, y_pred, metrics, config)
    logger.info("Run artifacts saved.")
    
    logger.info("Train and validation completed.")

    del label_values, y_pred, model

    return metrics

if __name__ == "__main__":
    main()
