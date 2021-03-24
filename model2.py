import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym
import network
import reward_model


class DynamicsModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, units=(32, 32)):
        super().__init__()

        # 隠れ層2層のMLP
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, units[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(units[0], units[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(units[1], output_dim)
        )

        self._loss_fn = torch.nn.MSELoss(reduction='mean')
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.to(device)
    def forward(self, inputs):
        #inputs = torch.cat([states, actions] , dim = -1)
        
        assert inputs.ndim == 2
        return self.model(inputs)

    def fit(self, inputs, labels):
        predicts = self.predict(inputs)
        loss = self._loss_fn(predicts, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.data.numpy()


env = gym.make('HalfCheetah-v2')
obs_dim = env.observation_space.high.size
act_dim = env.action_space.high.size

dynamics_model = DynamicsModel(input_dim=obs_dim + act_dim, output_dim=obs_dim)
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
        #inputs = np.concatenate([obs, action], axis=1)
        #inputs = torch.from_numpy(inputs).float()
        inputs = torch.cat([obs, action],dim=1) 
        dynamics_model(inputs)
        obs = obs_