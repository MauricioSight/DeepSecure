import json
from loggers.logging import Logger
import logging
from datasets.tow_ids_dataset import TOWIDSFeatureLoader
from execute_train_validation import main as execute_train_validation_main
from feature_generator.sliding_window_generator import SlidingWindowGenerator
from tuning.optimizer_factory import OptimizerFactory
from utils.config_handle import load_config
from utils.experiment_io import generate_run_id, get_run_dir, save_run_tune

def apply_trial_to_config(config: dict, trial_params: dict) -> dict:
    new_config = config.copy()

    for flat_key, value in trial_params.items():
        keys = flat_key.split("-")  # e.g., ['training', 'batch_size']
        current = new_config

        # Navigate to the parent dict
        for k in keys[:-1]:
            current = current.setdefault(k, {})

        # Overwrite the final key
        current[keys[-1]] = value

    return new_config

def main():
    # Load the configuration
    tune_config = load_config(default_file_name='tune_config')
    train_config = load_config(config_name=tune_config['train_config'], default_file_name='train_config')
    objective_metric = tune_config['objective_metric']

    if not hasattr(tune_config, 'run_id'):
        run_id = generate_run_id(model_name=tune_config['framework'], dataset_name=tune_config['algorithm'], 
                                phase=tune_config['train_config'])
        tune_config['run_id'] = run_id
    
    run_id = tune_config['run_id']
    run_dir = get_run_dir(run_id)

    tune_config['experiments_ids'] = []
    save_run_tune(run_dir, tune_config=tune_config, train_config=train_config)

    # Setup logger
    tune_logger = Logger(name="tune", log_file=f"{run_dir}/output.log", 
                    level=logging.DEBUG if hasattr(tune_config, 'debug') and tune_config['debug'] else logging.INFO)

    tune_logger.info("Starting tuning process...")
    tune_logger.info(f"[ RUN ID: {run_id} ]")

    # Load the dataset
    tune_logger.debug("Loading data...")
    feature_generator = SlidingWindowGenerator(train_config, tune_logger)
    dataset_loader = TOWIDSFeatureLoader(train_config, tune_logger, feature_generator)
    values, labels = dataset_loader.load_processed()
    tune_logger.info("Data loaded successfully.")

    # Initialize the optimizer
    tune_logger.info("Initializing optimizer...")

    def objective_fn(params):
        tune_logger.info(f"Trial parameters: ")
        tune_logger.info(json.dumps(params, indent=4))

        updated_config = apply_trial_to_config(train_config, params)
        tune_logger.debug(f"Updated config: {json.dumps(updated_config, indent=4)}")

        # Set the run ID for this trial
        experiment_id = generate_run_id(model_name=updated_config['model']['name'], 
                                        dataset_name=updated_config['dataset']['name'], phase=updated_config['phase'])
        updated_config['run_id'] = experiment_id

        # Save run_id
        tune_config['experiments_ids'].append(experiment_id)
        save_run_tune(run_dir, tune_config=tune_config, train_config=train_config)
        
        metrics = execute_train_validation_main(config=updated_config, values=values, labels=labels)

        tune_logger.info(f"Objective metric: {metrics[objective_metric]}")
        return metrics[objective_metric]
    
    optimizer_factory = OptimizerFactory(tune_config)
    optimizer = optimizer_factory.get_optimizer()
    optimizer.optimize(objective_fn)

    tune_logger.info("Tuning completed.")


if __name__ == "__main__":
    main()
