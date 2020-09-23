import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
import glob
import argparse

def convert_a2c_params(param):

    first_key = next(iter(param))

    # replace old learning_rate by new value of lr and rename
    param[first_key]['learning_rate'] = param[first_key].pop('lr')

    return param


def convert_acktr_params(param):

    first_key = next(iter(param))

    # replace old learning_rate by new value of lr and rename
    param[first_key]['learning_rate'] = param[first_key].pop('lr')

    return param


def convert_ddpg_params(param):

    first_key = next(iter(param))

    # replace old learning_rate by new value of lr and rename
    param[first_key]['actor_lr'] = param[first_key]['lr']
    param[first_key]['critic_lr'] = param[first_key]['lr']
    del param[first_key]['lr']

    # Apply layer normalization when using parameter perturbation
    if param[first_key]['noise_type'] == 'adaptive-param':
        param[first_key]['policy_kwargs'] = 'dict(layer_norm=True)'

    return param



def convert_ppo2_params(param):

    first_key = next(iter(param))

    if param[first_key]['n_steps'] < param[first_key]['batch_size']:
        param[first_key]['nminibatches'] = 1
    else:
        param[first_key]['nminibatches'] = int(param[first_key]['n_steps'] / param[first_key]['batch_size'])
    del param[first_key]['batch_size']

    # replace old learning_rate by new value of lr and rename
    param[first_key]['learning_rate'] = param[first_key].pop('lr')

    # replace old lambda by new value of lam and rename
    param[first_key]['lam'] = param[first_key].pop('lamdba')

    return param


def convert_sac_params(param):

    first_key = next(iter(param))

    # replace old learning_rate by new value of lr and rename
    param[first_key]['learning_rate'] = param[first_key].pop('lr')

    # create gradient step key
    param[first_key]['gradient_steps'] = param[first_key]['train_freq']

    # create policy_kwargs key
    if param[first_key]['net_arch'] == "small":
        param[first_key]['policy_kwargs'] = 'dict(layers=[64, 64])'
    elif param[first_key]['net_arch'] == "medium":
        param[first_key]['policy_kwargs'] = 'dict(layers=[256, 256])'
    elif param[first_key]['net_arch'] == "big":
        param[first_key]['policy_kwargs'] = 'dict(layers=[400, 300])'

    del param[first_key]['net_arch']

    return param


def convert_td3_params(param):

    first_key = next(iter(param))

    # replace old learning_rate by new value of lr and rename
    param[first_key]['learning_rate'] = param[first_key].pop('lr') 

    # create gradient step key
    param[first_key]['gradient_steps'] = param[first_key]['train_freq']

    # Apply layer normalization when using parameter perturbation
    if param[first_key]['noise_type'] == 'adaptive-param':
        param[first_key]['policy_kwargs'] = 'dict(layer_norm=True)'

    return param


def convert_trpo_params(param):

    first_key = next(iter(param))

    # replace old ent_coef by new value of entcoef and rename
    param[first_key]['entcoeff'] = param[first_key].pop('ent_coef')

    # replace old lambda by new value of lam and rename
    param[first_key]['lam'] = param[first_key].pop('lamdba')

    return param


def convert_her_params(param):
    first_key = next(iter(param))

    if param[first_key]['model_class'] == "sac":
        param = convert_sac_params(param)
    elif param[first_key]['model_class'] == "td3":
        param = convert_td3_params(param)
    elif param[first_key]['model_class'] == "ddpg":
        param = convert_ddpg_params(param)
    
    return param


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Log folder', type=str)
    args = parser.parse_args()
    log_dir = args.folder
    print(log_dir)


    # get final param.yml path
    filelist = []
    for path in Path(log_dir).rglob('*final_params.yml'):
        filelist.append(str(path))   # convert Posix path to list

    print(filelist)

    for filepath in filelist:

        print(filepath)

        # load params
        with open(filepath) as file:
            final_params = yaml.load(file, Loader=yaml.FullLoader)

        print("dirty params: ", final_params)

        if 'a2c' in filepath:
            print("A2C")
            cleaned_params = convert_a2c_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)

        elif 'acktr' in filepath:
            print("ACKTR")
            cleaned_params = convert_acktr_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)

        elif 'ddpg' in filepath:
            print("DDPG")
            cleaned_params = convert_ddpg_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)

        elif 'ppo2' in filepath:
            print("PPO2")
            cleaned_params = convert_ppo2_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)

        elif 'sac' in filepath:
            print("SAC")
            cleaned_params = convert_sac_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)
            

        elif 'td3' in filepath:
            print("TD3")
            cleaned_params = convert_td3_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)

        elif 'trpo' in filepath:
            print("TRPO")
            cleaned_params = convert_trpo_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)

        elif 'her' in filepath:
            print("HER")
            cleaned_params = convert_her_params(final_params)
            print("cleaned params: ", cleaned_params)
            with open(filepath[:-16]+'cleaned_params.yml', 'w') as f:
                yaml.dump(cleaned_params, f, default_flow_style=False)