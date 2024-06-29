import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions

import torch
import torch.nn as nn
import torch.distributions as distributions
from itertools import count, chain
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from rl_ppo.replay_buffer import Experience
import torch.nn.functional as F
from utils import save_rl_agent, load_rl_agent

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, use_tanh: bool):
        super(Actor, self).__init__()
      
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][use_tanh]
        # print("------use_orthogonal_init------")

        def orthogonal_init(layer, gain=1.0):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

  
    def forward(self, state_emb , selected_action_embs, training=True):
        """
        :param x: state embedding [N*D1]; A: pruning Action embeddings [N*K*D2]
        :return: v: action score [N*K]
        """

        s, A = state_emb, selected_action_embs
        s = s.unsqueeze(dim=1).repeat(1, A.size(1), 1)  # [N*D1] -> [N*K*D1]
        sa = torch.cat((s, A), dim=2)  # [N*K*(D1+D2)]
        sa = self.activate_func(self.fc1(sa))  # [N*K*D3]
        sa = self.activate_func(self.fc2(sa)) # [N*K*D3]
        a_prob = torch.softmax(self.fc3(sa).squeeze(dim=2), dim=1) # [N*K]
      
        action_dist = distributions.Categorical(probs=a_prob)
        if training: # sampling during training
            
            action_ind = action_dist.sample()   # shape: batch_size
            action_log_prob = action_dist.log_prob(action_ind)
            action_entropy = action_dist.entropy()
        else:
            action_log_prob = torch.zeros(a_prob.size(0), dtype=torch.float, device=state_emb.device)
            action_entropy = torch.zeros(a_prob.size(0), dtype=torch.float, device=state_emb.device)
            action_ind = torch.argmax(action_dist.probs, dim=-1)  # Deterministic action during testing/deployment
       
        return action_dist, action_ind, action_log_prob, action_entropy
    


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size, use_tanh: bool):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][use_tanh]

        # print("------use_orthogonal_init------")
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s
    


class Agent(object):
    def __init__(self, args, replay_buffer):
        self.state_size = args.state_size
        self.action_size = args.action_size
        self.hidden_size = args.hidden_size
        self.device = args.device

        self.buffer_size = args.buffer_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        # self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        
        self.actor = Actor(self.state_size, self.action_size, self.hidden_size, args.use_tanh).to(args.device)
        self.critic = Critic(self.state_size, self.hidden_size, use_tanh=args.use_tanh).to(args.device)
    

        self.replay_buffer = replay_buffer

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate,eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.learning_rate, eps=1e-5)


    def choose_action(self, state, action_embs, cand_action_inds, train=True, topk_item=10, topk_attr=10, rec_num=10, ask_num=2):
        """
        :param state: state_emb, shape: (1, state_emb_size)
        :param action_embs: shape: (1, topk_item+topk_attribute, action_emb_size)
        :param cand_action_inds: List[int]  length: topk_item+topk_attribute
        :param train: bool
        """

        
        with torch.no_grad():
            
            action_dist, action_ind, final_logprob, final_entropy = self.actor(state, action_embs, training=train)
           
            if action_ind.item() >= topk_item: # Ask action      
                ask_attribute = cand_action_inds[action_ind]   #
                valid_attribute_dist = action_dist.probs.squeeze(dim=0)[topk_item:]  # Attribute distribution
                topk_inds = torch.topk(valid_attribute_dist, ask_num)[1].tolist()
                selected_actions = [ask_attribute] + [cand_action_inds[idx+topk_item] for idx in topk_inds if idx!=action_ind.item()-topk_item]
                if len(selected_actions) > ask_num:
                    selected_actions = selected_actions[:ask_num]
                
                selected_actions = [action for action in selected_actions if action != -1]
                
                decison_ind = 1  # Ask action
            else:
                rec_item = cand_action_inds[action_ind]
                valid_items_dist = action_dist.probs.squeeze(dim=0)[:topk_item]
                selected_actions = []
                # 根据valid_items_dist 取TOPk self.k1-1个ind
                topk_inds = torch.topk(valid_items_dist, rec_num)[1].tolist()
                selected_actions = [rec_item] + [cand_action_inds[idx] for idx in topk_inds if idx!=action_ind.item()]
                decison_ind = 0  # Recommend action

            return decison_ind, selected_actions, final_logprob.item(), final_entropy.item()


        

    def update(self, total_steps):

        all_experiences = self.replay_buffer.get_all()

        # All data is batched
        user_id, state_emb, next_state_emb, a_logprob, action_embs, reward, done, rec_done =  Experience(*zip(*all_experiences))
        """
        state related : user_id, attribute_sequence_ids, next_attribute_sequence_ids  
        action related:  a_logprob
        actor input related: action_embs
        """

        user_id = torch.tensor(user_id, dtype=torch.int64).to(self.device)
        
        a_logprob = torch.tensor(a_logprob, dtype=torch.float).to(self.device).unsqueeze(dim=1) # shape: batch_size, 1
        action_embs = torch.stack(action_embs).to(self.device)

        reward = torch.tensor(reward, dtype=torch.float).to(self.device).unsqueeze(dim=1) # shape: batch_size, 1
        done = torch.tensor(done, dtype=torch.int64).to(self.device).unsqueeze(dim=1) # shape: batch_size, 1
        rec_done = torch.tensor(rec_done, dtype=torch.int64).to(self.device).unsqueeze(dim=1) # shape: batch_size, 1

        s = torch.stack(state_emb).to(self.device) # (bs, D+T+T+20)
        s_ = torch.stack(next_state_emb).to(self.device) # (bs, D+T+T+20)

        r = reward

 

        """
            Calculate the advantage using GAE
            'rec_ind>=0' means recommending success, there is no next state s'
            'done=True' represents the terminal of an episode(recommending success or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - rec_done) * vs_ - vs

            # to .numpy()
            deltas_cpu = deltas.to('cpu')
            done_cpu = done.to('cpu')

            gae = 0
            adv = []
            for delta, d in zip(reversed(deltas_cpu.flatten().numpy()), reversed(done_cpu.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)  # 
            v_target = adv + vs

            if self.use_adv_norm:  # Trick:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_size)), self.mini_batch_size, False):

                ## =============Actor input :  state + actions ================ 
                _, _, final_logprob, final_entropy = self.actor(s[index], action_embs[index], training=True)
                dist_entropy = final_entropy.view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = final_logprob.view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                # self.optimizer_sequence.zero_grad()

                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                
                
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                # self.optimizer_sequence.step()


        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        return actor_loss.mean().item(), critic_loss.mean().item(), dist_entropy.mean().item()

    def lr_decay(self, total_steps):
        lr_a_now = self.learning_rate * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_a_now
            
    def save_rl_model(self, data_name, filename, epoch_user):
       save_rl_agent(dataset=data_name, model={'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, filename=filename, epoch_user=epoch_user)

    def load_rl_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user)
        self.actor.load_state_dict(model_dict['actor'])
        self.critic.load_state_dict(model_dict['critic'])
            


