import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym
import network
import reward_model

def swish(x):
    return x*torch.sigmoid(x)

def predict_next_state_and_reward(state, action):
    idx = np.random.randint(0, n_models)
    mu, var = ensemble_models[idx](state, action)
    mu_r, var_r = ensemble_reward_models[idx](state, action)
    next_state = ensemble_models[idx].predict(mu, var)
    reward = ensemble_reward_models[idx].predict(mu_r, var_r)
    return next_state, reward



class Model(nn.Module):
    min_log_var = -5
    max_log_var = -1
    def __init__(self, state_shape, action_shape, buffer  ,reward_model, device , batch_size = 256, H_steps = 10, alpha = 0.0001):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_shape[0] + action_shape[0], 256)
        self.fc2 = nn.Linear(256,256)
        self.out = nn.Linear(256, state_shape[0] *2)
        self.net = nn.Sequential(self.fc1,
                                nn.ReLU(),
                                self.fc2,
                                nn.ReLU(),
                                self.out)
        
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.buffer = buffer
        self.batch_size = batch_size
        self.reward_model = reward_model
        self.H_steps = H_steps
        self.alpha = alpha
        #self.sac = sac
        self.device = device
        self.modeloptimizer = torch.optim.Adam(self.parameters() , lr = 3e-4)
        self.to(device)
    def forward(self, states, actions):
        
        x = torch.cat((states, actions), dim = -1)
        x = self.net(x)
        delta_mean , log_var = torch.chunk(x , 2, dim = 1)    
        log_var = torch.sigmoid(log_var)
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
        var = torch.exp(log_var)
        next_states_mean = delta_mean + states
        return next_states_mean, var

    def loss(self, states, actions, next_states_target):
        #print(actions.is_cuda)
        next_states, var = self.forward(states, actions)
        #print(next_states.shape)
        loss = (next_states - next_states_target) **2 /var + torch.log(var)
        loss = loss.mean()
        return loss

    def predict(self, mu , var):
        return Normal(mu , torch.sqrt(var)).sample()

    def generate_data(self):
        #D_model を定義
        model_buffer = buffer.ReplayBuffer(self.buff_size, state_shape, self.action_shape)
        #startするバッチをbufferから取り出す
        states, actions, *_ = self.buffer.sample_buffer(self.batch_size)
        #modelを用いてステップする
        for h in range(self.H_steps):
            for b in range(self.batch_size):
                action , _ = sac.explore(states[i])
                next_state, reward = predict_next_state(states[i], action)
                  
                model_buffer.add(states[i], action , next_state , reward , 0.)
                state[i] = next_state
        return model_buffer

    def update(self):
        losses = []
        reward_losses = []
        for states, actions, next_states, rewards in self.buffer.train_batchs(self.batch_size):
            self.modeloptimizer.zero_grad()
            #print(actions.shape)
            loss = self.loss(states, actions, next_states)
            #print(loss.is_cuda)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_value_(self.parameters() , 5)
            self.modeloptimizer.step()

            self.reward_model.modeloptimizer.zero_grad()
            reward_loss = self.reward_model.loss(states, actions, rewards)
            reward_loss.backward()
            reward_losses.append(reward_loss.item())
            torch.nn.utils.clip_grad_value_(self.reward_model.parameters() , 5)
            self.reward_model.modeloptimizer.step()
        loss = np.mean(losses)
        reward_loss = np.mean(reward_losses)
        print(loss, reward_loss)
        return loss,reward_loss



'''env = gym.make('HalfCheetah-v2')
SEED = 0
REWARD_SCALE = 5.0

NUM_STEPS = 10 ** 6
EVAL_INTERVAL = 10 ** 4



n_models = 5
algo = network.SAC(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    device = 'cuda:0',
    seed=SEED,
    reward_scale=REWARD_SCALE
)

reward_model = reward_model.Reward_Model(env.observation_space.shape, env.action_space.shape, device = 'cuda:0') 
model = Model(env.observation_space.shape, env.action_space.shape, buffer = algo.replay_buffer , reward_model = reward_model , device ='cuda:0')
for _ in range(1):
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs_, reward, done, _ = env.step(action)
        obs = np.expand_dims(obs, axis=0)
        action = np.expand_dims(action, axis=0)
        obs = torch.from_numpy(obs).float()
        action = torch.from_numpy(action).float()
        #print(model(obs, action))
        obs = obs_'''