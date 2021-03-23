import gym
import numpy as np
from network import SAC, Trainer
import sys

ENV_ID = 'HalfCheetah-v2'
SEED = 0
REWARD_SCALE = 5.0

NUM_STEPS = 10 ** 6
EVAL_INTERVAL = 10 ** 4

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

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)

trainer.train()