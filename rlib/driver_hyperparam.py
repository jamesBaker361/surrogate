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
from ray import tune
from ray.rllib.agents import sac, ppo, ddpg
from ray.rllib import agents

from envs.traffic_env_dir.traffic_env import TrafficEnv

import tensorflow as tf
from customModels import *

from ray.tune.logger import pretty_print

from datetime import datetime

now = datetime.now()

time_str=now.strftime("%m/%d-%H:%M:%S")

tf.config.run_functions_eagerly(False)


parser = argparse.ArgumentParser()
parser.add_argument(
	"--run", type=str, default="ppo", help="The RLlib-registered algorithm to use."
)

parser.add_argument(
    "--horizon", type=int, default=100, help="steps per episode"
)

parser.add_argument(
    "--beta",type=float, default=0.1, help="beta coefficient on regularization term"   
)

parser.add_argument(
    "--steps", type=int, default=1000,help="how many timesteps before stopping"
)

args = parser.parse_args()
print(f"Running with following CLI options: {args}")
print(args.run)

env_config={
			"demand": demand,
			"perturbed" :perturbed,
			"fake_flow" : fake_flow["flow"],
			"real_flow": real_flow["flow"],
			"horizon" : args.horizon,
            "fw_iterations":250,
            "reg_beta":args.beta
		}

trainer_type=args.run
timesteps_total=args.steps

def get_trial_function():
    def _trial_function(trial):
        return "hyperparam_{}_horizon={}_ts_total={}_beta={}_{}".format(args.run,args.horizon,args.steps,args.beta,time_str)
    return _trial_function

trial_name_creator=get_trial_function()

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
            "fcnet_hiddens":tune.grid_search([ [128 for _ in range(x)] for x in range(1,6) ])
        },
        "Q_model":{
            "fcnet_hiddens": tune.grid_search([ [128 for _ in range(x)] for x in range(1,6) ])
        },
        "env_config":env_config,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework":"tf2"
    }
    trainer=sac.SACTrainer

elif trainer_type=='td3':
    agent_config={
        "num_workers":4,
        'train_batch_size':500,
        "env_config":env_config,
        "actor_hiddens": tune.grid_search([ [128 for _ in range(x)] for x in range(1,6) ]),
        "critic_hiddens": tune.grid_search([ [128 for _ in range(x)] for x in range(1,6) ]),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework":"tf2"
    }
    trainer=ddpg.TD3Trainer

elif trainer_type=='impala':
    agent_config={"num_workers":4,
        'train_batch_size':500,
        "env_config":env_config,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "grad_clip": 1000.0, 
        "framework":"tf2"}
    trainer=agents.impala.ImpalaTrainer

print(dir(trainer))

status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

agent_config["env"]=TrafficEnv
stop={"timesteps_total":timesteps_total}

results = tune.run(trainer, 
    config=agent_config, stop=stop, verbose=1,
    trial_name_creator=trial_name_creator)