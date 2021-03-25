import gym
import numpy as np
from network import SAC, Trainer
import sys
from model import Model
from reward_model import Reward_Model


ENV_ID = 'HalfCheetah-v2'
SEED = 0
REWARD_SCALE = 5.0

NUM_STEPS = 50 * 10**3
EVAL_INTERVAL = 10 ** 3
MODEL_INTERVAL = 10 ** 3
n_models = 5
env = gym.make(ENV_ID)
env_test = gym.make(ENV_ID)

cuda = sys.argv[1]


algo = SAC(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    device = cuda,
    seed=SEED,
    reward_scale=REWARD_SCALE
)


ensemble_reward_models = [Reward_Model(env.observation_space.shape, env.action_space.shape, device = cuda) for _ in range(n_models) ]
ensemble_models = [Model(env.observation_space.shape, env.action_space.shape, buffer = algo.replay_buffer , reward_model = ensemble_reward_models[i], device = cuda) for i in range(n_models)] 
algo.ensemble_models = ensemble_models

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
  
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)

trainer.model_based_train()