import argparse

import yaml


def load_config(config=None, config_name=None):
    if config is not None:
        return config

    parser = argparse.ArgumentParser(description='Execute train validation step')
    parser.add_argument('--config', required=False, help='YAML File containing the configs')
    args = parser.parse_args()

    model_name = 'tune-SeqWatch' if args.config is None else args.config
    config_name = model_name if config_name is None else config_name
    with open(f"configs/{config_name}.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config

def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary, joining keys with sep.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items