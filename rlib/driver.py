
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import sys
import random

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

import tensorflow as tf
import frankwolfe as fw

sys.path.append('..')
from dataLoad import *


parser = argparse.ArgumentParser()
parser.add_argument(
	"--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
	"--framework",
	choices=["tf", "tf2", "tfe", "torch"],
	default="tf2",
	help="The DL framework specifier.",
)
parser.add_argument(
	"--as-test",
	action="store_true",
	help="Whether this script should be run as a test: --stop-reward must "
	"be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
	"--stop-iters", type=int, default=5, help="Number of iterations to train."
)
parser.add_argument(
	"--stop-timesteps", type=int, default=100, help="Number of timesteps to train."
)
parser.add_argument(
	"--stop-reward", type=float, default=-0.1, help="Reward at which we stop training."
)
parser.add_argument(
	"--no-tune",
	action="store_true",
	help="Run without Tune using a manual train loop instead. In this case,"
	"use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
	"--local-mode",
	action="store_true",
	help="Init Ray in local mode for easier debugging.",
)
"""
class MyEnv(gym.Env):
	def __init__(self, env_config):
		self.action_space = <gym.Space>
		self.observation_space = <gym.Space>
	def reset(self):
		return <obs>
	def step(self, action):
		return <obs>, <reward: float>, <done: bool>, <info: dict>"""

class TrafficEnvironment(gym.Env):
	"""Example of a custom env in for traffic"""

	def __init__(self, config: EnvContext):
		self.demand=config["demand"]
		self.initial_perturbed=config["perturbed"]
		self.perturbed=self.initial_perturbed.copy()
		self.initial_state=config["fake_flow"] #[f0,f1,f2...]
		self.state=self.initial_state.copy()
		self.real_flow=config["real_flow"] #[f0,f1,f2...]
		self.edges_count=len(self.state)

		self.episode_ended = False
		self.step_count=0
		self.horizon=config['horizon']

		space_size=len(self.state)

		self.action_space = Box(np.array([0.0 for _ in range(space_size)]),np.array([1.0 for _ in range(space_size)]))
		self.observation_space =Box(np.array([0.0 for _ in range(space_size)]),np.array([sys.maxsize for _ in range(space_size)]))
		

	def reset(self):
		self.state = self.initial_state.copy()
		self.perturbed=self.initial_perturbed.copy()
		self.episode_ended = False
		self.step_count=0
		return [self.state]

	def step(self, action):
		self.step_count+=1
		self.episode_ended = self.step_count >= self.horizon

		'''pairs=fw.importODPairsFrom(self.demand)
		graph=fw.Graph(self.perturbed,0,100)
		assign=fw.FrankWolfeAssignment(graph,pairs,True,False)

		self.state=assign.runPython(100)'''
		atp=fw.AssignTrafficPython()
		self.state=atp.flow(self.demand,self.perturbed,100)
		diff=[]
		for a,b in zip(self.real_flow["flow"],self._state):
			diff.append(np.abs(a-b))
		reward=-np.linalg.norm(diff)
		if self.episode_ended is False:
			for x in range(len(action)):
				self.perturbed["capacity"][x]=int(750*action[x])+250
		return [self.state], reward, self.episode_ended, {}

class EmptyEnvironment(gym.Env):
	def __init__(self,config: EnvContext):
		self.action_space = Discrete(2)
		self.observation_space = Box(0.0, 20, shape=(1,), dtype=np.float32)
		self.state=0

	def reset(self):
		self.state =0
		return [self.state]	

	def step(self,action):
		return [self.state], 10, True,{}

class CustomModel(TFModelV2):
	"""Example of a keras custom model that just delegates to an fc-net."""

	def __init__(self, obs_space, action_space, num_outputs, model_config, name):
		super(CustomModel, self).__init__(
			obs_space, action_space, num_outputs, model_config, name
		)
		self.model = FullyConnectedNetwork(
			obs_space, action_space, num_outputs, model_config, name
		)

	def forward(self, input_dict, state, seq_lens):
		return self.model.forward(input_dict, state, seq_lens)

	def value_function(self):
		return self.model.value_function()

if __name__ == "__main__":
    '''
	args = parser.parse_args()
	print(f"Running with following CLI options: {args}")

	ray.init(local_mode=args.local_mode)

	# Can also register the env creator function explicitly with:
	# register_env("corridor", lambda config: SimpleCorridor(config))
	ModelCatalog.register_custom_model(
		"my_model", CustomModel
	)

	config = {
		"env": TrafficEnvironment,  # or "corridor" if registered above
		"env_config": {
			"demand": demand,
			"perturbed" :perturbed,
			"fake_flow" : fake_flow["flow"],
			"real_flow": real_flow["flow"],
			"horizon" : 100
		},
		# Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
		"num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
		"model": {
			"custom_model": "my_model",
			"vf_share_layers": True,
		},
		"num_workers": 1,  # parallelism
		"framework": args.framework,
	}

	stop = {
		"training_iteration": args.stop_iters,
		"timesteps_total": args.stop_timesteps,
		"episode_reward_mean": args.stop_reward,
	}

	if args.no_tune:
		# manual training with train loop using PPO and fixed learning rate
		if args.run != "PPO":
			raise ValueError("Only support --run PPO with --no-tune.")
		print("Running manual train loop without Ray Tune.")
		ppo_config = ppo.DEFAULT_CONFIG.copy()
		ppo_config.update(config)
		# use fixed learning rate instead of grid search (needs tune)
		ppo_config["lr"] = 1e-3
		trainer = ppo.PPOTrainer(config=ppo_config, env=TrafficEnvironment)
		# run manual training loop and print results after each iteration
		for _ in range(args.stop_iters):
			result = trainer.train()
			print(pretty_print(result))
			# stop training of the target train steps or reward are reached
			if (
				result["timesteps_total"] >= args.stop_timesteps
				or result["episode_reward_mean"] >= args.stop_reward
			):
				break
	else:
		# automated run with Tune and grid search and TensorBoard
		print("Training automatically with Ray Tune")
		results = tune.run(args.run, config=config, stop=stop)

		if args.as_test:
			print("Checking if learning goals were achieved")
			check_learning_achieved(results, args.stop_reward)

	ray.shutdown()'''