import sys
sys.path.append('..')
from qlearn.dataLoad import *

import gym
import envs
import argparse
import os

import frankwolfe as fw
import unittest
import gc
import pickle
from random import randint, choice,sample
import ray
from ray.rllib.agents import sac, ppo, ddpg
from ray.rllib import agents

from envs.traffic_env_dir.traffic_env import TrafficEnv

import tensorflow as tf
from customModels import *

from ray.tune.logger import pretty_print

tf.config.run_functions_eagerly(False)

env_config={
			"demand": demand,
			"perturbed" :perturbed,
			"fake_flow" : fake_flow["flow"],
			"real_flow": real_flow["flow"],
			"horizon" : 25
		}

parser = argparse.ArgumentParser()
parser.add_argument(
	"--run", type=str, default="ppo", help="The RLlib-registered algorithm to use."
)

args = parser.parse_args()
print(f"Running with following CLI options: {args}")
print(args.run)

trainer_type=args.run

if trainer_type=='ppo':

    agent_config = {'gamma': 0.9,
            'lr': 1e-2,
            'num_workers': 4,
            'train_batch_size': 500,
            'vf_clip_param': 100000,
            'model': {
                'fcnet_hiddens': [256,256,256,256]
            },
            "env_config":env_config,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework":"tf2"}
    trainer = ppo.PPOTrainer(env=TrafficEnv, config=agent_config)

elif trainer_type == "sac":
    agent_config={
        'train_batch_size':500,
        'num_workers':4,
        "policy_model": {
            "custom_model": BasicModel,
            "custom_model_config": {"fcnet_hiddens":[256,256, 256,256]}
        },
        "env_config":env_config,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework":"tf2"
    }
    trainer=sac.SACTrainer(env=TrafficEnv,config=agent_config)

elif trainer_type=='td3':
    agent_config={
        "num_workers":4,
        'train_batch_size':500,
        "env_config":env_config,
        "actor_hiddens": [400,400, 300,300],
        "critic_hiddens": [400,400, 300,300],
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework":"tf2"
    }
    trainer=ddpg.TD3Trainer(env=TrafficEnv, config=agent_config)

elif trainer_type=='impala':
    agent_config={"num_workers":4,
        'train_batch_size':500,
        "env_config":env_config,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "grad_clip": 1000.0, 
        "framework":"tf2"}
    trainer=agents.impala.ImpalaTrainer(env=TrafficEnv, config=agent_config)

print(dir(trainer))

status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

max_training_episodes = 1000
for i in range(max_training_episodes):
    result = trainer.train()
    if i%1 == 0:
        checkpoint = 'no checkpoint' #trainer.save('/global/homes/j/jamesbak/m1248/rlib/check_{}'.format(trainer_type))
        print(status.format(
            i + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            checkpoint
            ))