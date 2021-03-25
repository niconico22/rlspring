import torch 
import torch.nn as nn
import numpy as np
import math
import buffer
from time import time
from abc import ABC, abstractmethod

import logging 
import datetime

def set_log(s):
    # ログレベルを DEBUG に変更
    now = datetime.datetime.now()
    filename = './' + s + 'log/' + 'log_' + \
        now.strftime('%Y%m%d_%H%M%S') + '.log'
    
    formatter = '%(levelname)s : %(asctime)s : %(message)s'

    logging.basicConfig(filename=filename,
                        level=logging.DEBUG, format=formatter)





n_models = 5
H_steps = 1

def predict_next_state_and_reward(state, action, ensemble_models, device):
    state = torch.from_numpy(state).float().to(device)
    state = state.unsqueeze_(0)
    action = torch.from_numpy(action).float().to(device)
    action = action.unsqueeze_(0)
    idx = np.random.randint(0, n_models)
    mu, var = ensemble_models[idx](state, action)
    mu_r, var_r = ensemble_models[idx].reward_model(state, action)
    next_state = ensemble_models[idx].predict(mu, var)
    reward = ensemble_models[idx].reward_model.predict(mu_r, var_r)
    return next_state, reward




def caluculate_log_pi( log_stds , noises, actions):
    gaussian_log_probs = (-0.5* noises.pow(2) - log_stds).sum(dim=-1, keepdim=True)\
        -0.5*math.log(2*math.pi)*log_stds.size(-1)
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6 ).sum(dim=-1, keepdim =True)

    return log_pis
    

def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    us = means + stds*noises
    actions = torch.tanh(us)
    log_pis = caluculate_log_pi(log_stds, noises, actions)
    return actions, log_pis
    


class Critic_network(nn.Module):
    def __init__(self,state_shape,action_shape):
        super().__init__()
        self.fc1 = nn.Linear(state_shape[0]+ action_shape[0], 256 )
        self.fc2 = nn.Linear(256,256)
        self.output1 = nn.Linear(256,1)

        self.fc3 = nn.Linear(state_shape[0]+ action_shape[0], 256 )
        self.fc4 = nn.Linear(256,256)
        self.output2 = nn.Linear(256,1)

        self.net1 = nn.Sequential(self.fc1, 
                    nn.ReLU(),
                    self.fc2,
                    nn.ReLU(),
                    self.output1)


        self.net2 = nn.Sequential(self.fc3, 
                    nn.ReLU(),
                    self.fc4,
                    nn.ReLU(),
                    self.output2)
        
    def forward(self,states , actions):
        q1 = self.net1(torch.cat([states, actions] , dim = -1))
        
        q2 = self.net2(torch.cat([states, actions] , dim = -1))
        
        return q1, q2
    
    

class Actor_network(nn.Module):
    def __init__(self,state_shape, action_shape):
        super().__init__()

        self.fc1 = nn.Linear(state_shape[0], 256 )
        self.fc2 = nn.Linear(256,256)
        self.output = nn.Linear(256,action_shape[0] * 2)
        self.net = nn.Sequential(self.fc1, 
                                nn.ReLU(),
                                self.fc2,
                                nn.ReLU(),
                                self.output)
    def forward(self,states):
        actions = self.net(states).chunk(2,dim=-1)[0]
        actions = torch.tanh(actions)
        return actions

    def sample(self,states):
        means, log_stds = self.net(states).chunk(2,dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))
    
    

class Algorithm(ABC):

    def explore(self, state):
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        """ 決定論的な行動を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, steps):
        """ 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す． """
        pass

    @abstractmethod
    def step(self, env, state, t, steps):
        """ 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            受け取り，リプレイバッファへの保存などの処理を行い，状態・エピソードのステップ数を更新する．
        """
        pass

    @abstractmethod
    def update(self):
        """ 1回分の学習を行う． """
        pass

        

class SAC(Algorithm):
    def __init__(self, state_shape, action_shape,  device, ensemble_models = None, seed = 0,
                batch_size=256, gamma = 0.99 , lr=3e-4,alpha = 0.2, buff_size = 10**6, start_steps =2*10**3, tau =5e-3, reward_scale = 1.0):
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.replay_buffer = buffer.ReplayBuffer(buff_size=buff_size, state_shape=state_shape, action_shape = action_shape, device = device)
        self.actor = Actor_network(state_shape = state_shape, action_shape = action_shape).to(device)
        self.critic = Critic_network(state_shape = state_shape, action_shape = action_shape).to(device)
        self.critic_target = Critic_network(state_shape = state_shape, action_shape = action_shape).to(device).eval()

        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.learning_steps = 0
        self.device = device
        self.gamma = gamma 
        self.lr = lr
        self.buff_size = buff_size
        self.start_steps = start_steps
        self.tau = tau 
        self.alpha = alpha
        self.reward_scale =reward_scale
        self.ensemble_models = ensemble_models

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)
    
    def update(self):
        #print(self.learning_steps)
        self.learning_steps += 1
        #states, actions, next_states, rewards, dones = self.replay_buffer.sample_buffer(self.batch_size)
        #modelからデータを生成
        model_buffer = self.generate_data(self.replay_buffer)
        #real and model data　のサンプルを取る
        sz = np.random.randint(low = 0 , high = self.replay_buffer.n + model_buffer.n , size = self.batch_size)
        real_batch_size = np.sum(sz < self.replay_buffer.n)
        model_batch_size = self.batch_size - real_batch_size
        states, actions, next_states, rewards, dones = self.replay_buffer.sample_buffer(real_batch_size)
        model_states, model_actions, model_next_states, model_rewards, model_dones = model_buffer.sample_buffer(model_batch_size)
        
        states = torch.cat([states, model_states] , dim = 0 )
        actions = torch.cat([actions, model_actions] , dim = 0)
        next_states = torch.cat([next_states, model_next_states] , dim = 0 )
        rewards = torch.cat([rewards , model_rewards], dim = 0)
        dones = torch.cat([dones, model_dones] , dim = 0)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()


    def update_sac(self):
        states, actions, next_states, rewards, dones = self.replay_buffer.sample_buffer(self.batch_size)
        
        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs
        #print(next_actions[0])
        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()
        #print(log_pis[0])
        self.critic_optimizer.zero_grad()
        #print(loss_critic1)
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.critic_optimizer.step()

    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.actor_optimizer.step()

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    

    def step(self, env, state, t, steps):
        t += 1

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する．
        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            action, _ = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        # リプレイバッファにデータを追加する．
        self.replay_buffer.add(state, action, next_state,reward, done_masked)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def generate_data(self, replay_buffer):
        #D_model を定義
        model_buffer = buffer.ReplayBuffer(replay_buffer.buff_size, replay_buffer.state_shape, replay_buffer.action_shape, device = self.device)
        #startするバッチをbufferから取り出す
        states,*_ = replay_buffer.sample_buffer(100)
        states = states.cpu().numpy()
        #modelを用いてステップする
        for h in range(H_steps):
            for b in range(100):
                action , _ = self.explore(states[b])
                next_state, reward = predict_next_state_and_reward(states[b], action, self.ensemble_models, self.device)
                next_state = next_state.cpu().numpy()[0]
                reward = reward.cpu().numpy()  
                model_buffer.add(states[b], action , next_state , reward , 0.)
                states[b] = next_state
        return model_buffer



class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=3, model_interval = 10**3):

        self.env = env
        self.env_test = env_test
        self.algo = algo
        #アンサンブルモデルを追加
        # 環境の乱数シードを設定する．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': []}

        # データ収集を行うステップ数．
        self.num_steps = num_steps
        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes

        self.model_interval = model_interval
        set_log('data')
        logging.info("H_steps %d algo_updata: %d" , H_steps, 5)
    def train(self):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
            state, t = self.algo.step(self.env, state, t, steps)

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                
                self.algo.update_sac()

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0:
                self.evaluate(steps)

    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        returns = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              )
        logging.info('Num steps: %d Return: %.1f' , steps, mean_return)

    def model_based_train(self):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
            state, t = self.algo.step(self.env, state, t, steps)
            if steps % 100 == 0:
                print(steps)
            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                for i in range(5):    
                    self.algo.update()

            if steps % self.model_interval == 0 :
                
                for step in range(10):
                    for i in range(5):
                        self.algo.ensemble_models[i].update()
            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0:
                self.evaluate(steps)
    
    