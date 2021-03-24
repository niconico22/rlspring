import numpy as np
import torch 
import torch.nn as nn
from torch.distributions import Normal
import gym
import network


def swish(x):
    return x*torch.sigmoid(x)

def predict_next_state(state, action):
    idx = np.random.randint(0, n_models)
    mu, var = ensemble_models[idx](state, action)
    next_state = ensemble_models[idx].predict(mu, var)
    return next_state


class Reward_Model(nn.Module):
    min_log_var = -5
    max_log_var = -1
    def __init__(self,state_shape, action_shape,  device):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.fc1 = nn.Linear(self.state_shape[0] + self.action_shape[0], 256)
        self.fc2 = nn.Linear(256,256)
        self.out = nn.Linear(256, 2)
        self.net = nn.Sequential(self.fc1,
                                nn.ReLU(),
                                self.fc2,
                                nn.ReLU(),
                                self.out)
        self.modeloptimizer = torch.optim.Adam(self.parameters() , lr = 3e-4)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim = 1)
        x = self.net(x)
        rewards_mean , log_var = torch.chunk(x , 2, dim = 1)    
        log_var = torch.sigmoid(log_var)
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
        var = torch.exp(log_var)
        return rewards_mean, var

    def loss(self, states, actions, rewards_target):
        rewards, var = self(states, actions)
        loss = (rewards - rewards_target) **2 /var + torch.log(var)
        loss = loss.mean()
        return loss

    def predict(self, mu , var):
        return Normal(mu , torch.sqrt(var)).sample()

