import numpy as np
import torch
import time
import torch.distributed as dist

import numpy as np
import torch

import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, dataloader=None):

        train_losses = []
        mask_ave_all_1=[]
        mask_ave_all_2=[]
        mask_ave_all_3=[]
        logs = dict()

        train_start = time.time()

        self.model.train()
        if self.scheduler is not None:
            self.scheduler.step()

        for _, trajs in enumerate(dataloader):

            train_loss,mask_ave = self.train_step(trajs,device=self.model.device)

            train_losses.append(train_loss)

            mask_ave_all_1.append(mask_ave[0])
            mask_ave_all_2.append(mask_ave[1])
            mask_ave_all_3.append(mask_ave[2])

        mask_ave_all_1_cpu = [m.cpu().numpy() for m in mask_ave_all_1]
        mask_ave_all_2_cpu = [m.cpu().numpy() for m in mask_ave_all_2]
        mask_ave_all_3_cpu = [m.cpu().numpy() for m in mask_ave_all_3]

        logs['mask_ave_all_1'] = np.mean(mask_ave_all_1_cpu)
        logs['mask_ave_all_2'] = np.mean(mask_ave_all_2_cpu)
        logs['mask_ave_all_3'] = np.mean(mask_ave_all_3_cpu)


        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def save_model(self, env_name, iter, folder):
        model_name = env_name + '/' + str(iter) + '.pt'
        torch.save(self.model.state_dict(),folder+model_name)  # model save
        print('model saved to ', folder+model_name)


    

class DecisionTransformerTrainer(Trainer):

    def train_step(self,trajs,device=None):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = trajs

        states=states.to(device)

        actions=actions.to(device)
        rewards=rewards.to(device)
        dones=dones.to(device)
        rtg=rtg.to(device)
        timesteps=timesteps.to(device)
        attention_mask=attention_mask.to(device)


        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds, mask_ave= self.model.forward(
            states, actions, rtg[:,:-1], timesteps, attention_mask=attention_mask
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()





        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(),mask_ave
    


















# class PromptSequenceTrainer:

#     def __init__(self, model, optimizer, loss_fn,scheduler=None, eval_fns=None,variant=None, train_env_name_list=None):
#         self.model = model
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.scheduler = scheduler
#         self.eval_fns = [] if eval_fns is None else eval_fns
#         self.diagnostics = dict()
#         self.variant = variant
#         self.train_env_name_list = train_env_name_list




#         if dist.get_rank() == 0:
#             print('whole iteration',i)
            
#         logs['time/training'] = time.time() - train_start
#         logs['training/train_loss_mean'] = np.mean(train_losses)
#         logs['training/train_loss_std'] = np.std(train_losses)

#         for k in self.diagnostics:
#             logs[k] = self.diagnostics[k]

#         return logs

#     def train_step_mix(self,trajs):

#         states_list = []
#         actions_list = []
#         rewards_list = []
#         rtg_list = []
#         timesteps_list = []
#         attention_mask_list = []

#         for t,i in enumerate(self.train_env_name_list):

#             states=trajs[i]['s'].squeeze(0)
#             actions=trajs[i]['a'].squeeze(0)
#             rewards=trajs[i]['r'].squeeze(0)
#             rtg=trajs[i]['rtg'].squeeze(0)
#             timesteps=trajs[i]['timesteps'].squeeze(0)
#             attention_mask=trajs[i]['mask'].squeeze(0)

#             states_list.append(states)
#             actions_list.append(actions)
#             rewards_list.append(rewards)
#             rtg_list.append(rtg)
#             timesteps_list.append(timesteps)
#             attention_mask_list.append(attention_mask)

#         states = torch.cat(states_list, dim=0)
#         actions = torch.cat(actions_list, dim=0)
#         rewards = torch.cat(rewards_list, dim=0)
#         rtg = torch.cat(rtg_list, dim=0)
#         timesteps = torch.cat(timesteps_list, dim=0)
#         attention_mask = torch.cat(attention_mask_list, dim=0)


#         loss = self.model.forward(states=states,actions=actions ,rewards=rewards, returns_to_go=rtg[:,:-1] ,timesteps=timesteps , attention_mask=attention_mask, device=self.model.device,loss_fn=self.loss_fn)
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
#         self.optimizer.step()

#         return loss.detach().cpu().item()
