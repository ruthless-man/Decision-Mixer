import torch
import numpy as np
import gym
import os
import random
import matplotlib.pyplot as plt

from DM.DM_decision_transformer import DecisionTransformer


from DM.DM_trainer import DecisionTransformerTrainer

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device='cuda')
    return x

def get_env_info(env_name, dataset):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 7200, 36000, 72000]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 24000, 120000, 240000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 10000, 50000, 100000]
        scale = 1000.
    elif env_name == 'antmaze':
        import d4rl
        env = gym.make(f'{env_name}-{dataset}-v2')
        max_ep_len = 1000
        env_targets = [1.0, 10.0, 100.0, 1000.0, 100000.0] # successful trajectories have returns of 1, unsuccessful have returns of 0
        scale = 1.
    elif env_name == 'maze2d':
        if 'open' in dataset:
            dversion = 0
        else:
            dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [300, 200, 150,  100, 50, 20]
        scale = 10.
    elif env_name == 'kitchen':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [500, 250]
        scale = 100.
    elif env_name == 'pen':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'hammer':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 1000.
    elif env_name == 'door':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [2000, 1000, 500]
        scale = 100.
    elif env_name == 'relocate':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3000, 1000]
        scale = 1000.
        dversion = 1
    else:
        raise NotImplementedError
    
    return env, max_ep_len, env_targets, scale

def get_model_optimizer(variant, state_dim, act_dim, returns, scale, K, max_ep_len, device):
    if variant['model_type'] == 'dc':
        print('Please Using DecisionTransformer')
    elif variant['model_type'] == 'dt':
        model = DecisionTransformer(
            env_name=variant['env'],
            dataset=variant['dataset'],
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout']
        )
    else:
        raise NotImplementedError
    model = model.to(device=device)
    
    warmup_steps = variant['warmup_steps']


    def get_parameters_excluded_by_name(model, excluded_names):
        excluded_params = set()
        
        for name, param in model.named_parameters():
            if any(excluded_name in name for excluded_name in excluded_names):
                excluded_params.add(param)
        
        return excluded_params
 
    excluded_params = get_parameters_excluded_by_name(model,{'aux_predictor'})




    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad and p not in excluded_params, model.parameters()),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )




    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    return model, optimizer, scheduler




def get_trainer(model_type, **kwargs):
    if model_type == 'dc':
        return DecisionConvFormerTrainer(**kwargs)
    elif model_type == 'dt':
        return DecisionTransformerTrainer(**kwargs)
    
def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path