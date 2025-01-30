import json, pickle, os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import wandb

import argparse
import pickle


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
import numpy as np


import random
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from DM.DM_utils import discount_cumsum, torchify, get_env_info, get_model_optimizer, get_trainer

from DM.DM_evaluate_episodes import evaluate_episode_rtg

from d4rl import infos


class RLData(Dataset):
    def __init__(self,batch_size,max_len, trajectories, sorted_inds, scale, state_dim, act_dim, state_mean, state_std, max_ep_len, p_sample,device):

        num_trajectories = len(trajectories)
        data_dict = {}

        batch_inds = np.concatenate([
            np.random.choice(
                np.arange(num_trajectories),
                size=64,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )
            for _ in range(100)  # Rep
        ])


        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(64*100):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        
        # 构建 data_dict
        data_dict['s'] = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32)
        data_dict['a'] = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32)
        data_dict['r'] = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32)
        data_dict['d'] = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long)
        data_dict['rtg'] = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32)
        data_dict['timesteps'] = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long)
        data_dict['mask'] = torch.from_numpy(np.concatenate(mask, axis=0))
        
        self.data_dict = data_dict 

    def __len__(self):
        return len(self.data_dict['s'])

    def __getitem__(self, idx):
        s = self.data_dict['s'][idx]
        a = self.data_dict['a'][idx]
        r = self.data_dict['r'][idx]
        d = self.data_dict['d'][idx]
        rtg = self.data_dict['rtg'][idx]
        timesteps = self.data_dict['timesteps'][idx]
        mask = self.data_dict['mask'][idx]
        return (s, a, r, d, rtg, timesteps, mask)




def run(variant):

    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{env_name}-{dataset}'

    env, max_ep_len, env_targets, scale = get_env_info(env_name, dataset)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    if env_name=='kitchen':
        dataset_path = f'data-gym/{env_name}-{dataset}-v0.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif env_name=='maze2d' or env_name=='pen' or env_name=='hammer' or env_name=='door':
        dataset_path = f'data-gym/{env_name}-{dataset}-v1.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif env_name=='antmaze':
        dataset_path = f'data-gym/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'medium-expert':
        dataset_path = f'data-gym/{env_name}-expert-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        dataset_path = f'data-gym/{env_name}-medium-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories += pickle.load(f)
        random.shuffle(trajectories)
    else:
        dataset_path = f'data-gym/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed': 
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)


    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')

    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)


    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)


    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]


    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    prefix = variant['env'] + "_" + variant['dataset']




    def eval_episodes(target):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device
                    )
                returns.append(ret)
                lengths.append(length)

            reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v2"]
            reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v2"]
            print('reward_min',reward_min)
            print('reward_max',reward_max)

            return {
                # f'target_{target}_return_mean': np.mean(returns),
                # f'target_{target}_return_std': np.std(returns),
                # f'target_{target}_length_mean': np.mean(lengths),
                f'target_{target}_d4rl_score': (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),
            }
        return fn

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')


    model, optimizer, scheduler = get_model_optimizer(variant, state_dim, act_dim, returns, scale, K, max_ep_len, device)
    loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)


    model = model.to(device)
    model = DDP(model, device_ids=[device_id],find_unused_parameters=True)

    trainer = get_trainer(
        model_type=variant['model_type'],
        model=model,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )

    if dist.get_rank() == 0:
        pbar = tqdm(range(variant['max_iters']), disable=False)
    else:
        pbar = range(variant['max_iters'])  


    for iter in pbar:
            
        # trainer.save_model(
        #         env_name=variant['env']+variant['dataset'], 
        #         iter=iter, 
        #         folder='/data1/users/zhenghongling/Mixture-of-depths/model_saved/')
            
        datasets = RLData(variant['batch_size'],variant['K'], trajectories, sorted_inds, scale, state_dim, act_dim, state_mean, state_std, max_ep_len, p_sample,device)
        sampler = DistributedSampler(datasets, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        dataloader = DataLoader(datasets, batch_size=variant['batch_size'], sampler=sampler, num_workers=variant['num_workers'])
            
        dataloader.sampler.set_epoch(iter)
        outputs = trainer.train_iteration(dataloader=dataloader)



        
        if iter % variant['test_interval'] == 0:
            model.eval()
            for eval_fn in [eval_episodes(tar) for tar in env_targets]:
                eval_outputs= eval_fn(model)
                for k, v in eval_outputs.items():
                    outputs[f'evaluation/{k}'] = v

            if dist.get_rank() == 0:
                print('=' * 80)
                for k, v in outputs.items():
                    print(f'{k}: {v}')


        # if iter % variant['save_interval'] == 0:
        #     trainer.save_model(
        #         env_name=variant['env']+variant['dataset'], 
        #         iter=iter, 
        #         folder='/data1/users/zhenghongling/Mixture-of-depths/model_saved/')
            

        if dist.get_rank() == 0:
            outputs.update({"global_step": iter})
            pbar.set_description(f"Iteration {iter}")
        
        if variant['log_to_wandb']:
            wandb.log(outputs)

# torchrun --nproc_per_node=1 --master_port=25614 main.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah') # walker2d, hopper, halfcheetah,antmaze,hammer,pen,door,relocate,kitchen,maze2d
    parser.add_argument('--dataset', type=str, default='medium-replay')  # medium,umaze, umaze-diverse,medium-replay, medium-expert, expert,cloned,complete,partial
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse, no-reward-decay
    parser.add_argument('--K', type=int, default=120)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='dt')  
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=5)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--remove_act_embs', action='store_true')
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)


    args = parser.parse_args()

    run(variant=vars(args))


