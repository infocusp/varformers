import pandas as pd
import yaml
import json

# TODO: Merge all into single function for read and dump
def read_yaml(filepath):
    """This function returns config file loaded from yaml."""
    with open(filepath, 'r') as fh:
        config = yaml.safe_load(fh)
    return config

def read_json(filepath):
    """This function returns json file object"""
    with open(filepath, 'r') as fh:
        config = json.load(fh)
    return config

def dump_yaml(config, filepath):
    """This function stores config file to filepath"""
    with open(filepath, 'w') as fh:
        config = yaml.dump(config, fh)
    return

def dump_json(config, filepath):
    """This function stores json file to filepath"""
    with open(filepath, 'w') as fh:
        config = json.dump(config, fh, indent=2)
    return
    
def read_data(filepath, backed='r'):
    """This function reads the anndata in backed `r mode."""
    if '.csv' in filepath:
        data = pd.read_csv(filepath)
    else:
        raise ValueError('Only .csv files supported!!!')
    return data


