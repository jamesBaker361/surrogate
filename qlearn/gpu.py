import abc
import tensorflow as tf
import numpy as np

import reverb

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.agents.sac import sac_agent
from tf_agents.metrics import py_metrics
from tf_agents.trajectories import time_step as ts

from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

import multiprocessing as mp

from dataLoad import *
from TrafficEnv import TrafficEnvironment

from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network

from networks import *

from tqdm import tqdm

if __name__=="__main__":
	pert_indices=[x for x in range(len(labels["label"])) if labels["label"][x]==0]
	environment = TrafficEnvironment(demand,edges,perturbed,fake_flow,real_flow,pert_indices)
	collect_env =TrafficEnvironment(demand,edges,perturbed,fake_flow,real_flow,pert_indices)
	eval_env=TrafficEnvironment(demand,edges,perturbed,fake_flow,real_flow,pert_indices)
	#utils.validate_py_environment(environment, episodes=5)
	tf_env=tf_py_environment.TFPyEnvironment(environment)
	edges_count=len(fake_flow["flow"])
	action_spec = array_spec.BoundedArraySpec(shape=(edges_count,), dtype=np.float32, minimum=0, maximum=1, name='action')
	observation_spec = array_spec.BoundedArraySpec(shape=(edges_count,), dtype=np.float32, minimum=0, name='observation')

	observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))

	replay_buffer_capacity = 10000
	actor_fc_layers=(256,256)
	critic_action_fc_layers=None
	critic_joint_fc_layers=(256,256)
	critic_obs_fc_layers=None
	initial_collect_steps = 100
	num_eval_episodes=10

	num_iterations = 100000 # @param {type:"integer"}

	initial_collect_steps = 10000 # @param {type:"integer"}
	collect_steps_per_iteration = 1 # @param {type:"integer"}
	replay_buffer_capacity = 10000 # @param {type:"integer"}

	batch_size = 256 # @param {type:"integer"}

	critic_learning_rate = 3e-4 # @param {type:"number"}
	actor_learning_rate = 3e-4 # @param {type:"number"}
	alpha_learning_rate = 3e-4 # @param {type:"number"}
	target_update_tau = 0.005 # @param {type:"number"}
	target_update_period = 1 # @param {type:"number"}
	gamma = 0.99 # @param {type:"number"}
	reward_scale_factor = 1.0 # @param {type:"number"}

	actor_fc_layer_params = (256, 256)
	critic_joint_fc_layer_params = (256, 256)

	log_interval = 5 # @param {type:"integer"}

	num_eval_episodes = 10 # @param {type:"integer"}
	eval_interval = 10000 # @param {type:"integer"}

	policy_save_interval = 5000 # @param {type:"integer"}

	use_gpu = True

	print("gpu is ",use_gpu)
	agent_variety="sac"
	print("agent type =",agent_variety)
	tempdir="dirgpu{}{}/".format(use_gpu,agent_variety)


	strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
	with strategy.scope():
		actor_net = create_actor_network(actor_fc_layers, tf_env.action_spec())
		critic_net = create_critic_network(critic_obs_fc_layers,
										critic_action_fc_layers,
										critic_joint_fc_layers)
	actor_learning_rate=1e-4
	critic_learning_rate=1e-3

	saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)
	collect_dir=os.path.join(tempdir, learner.TRAIN_DIR)
	eval_dir=os.path.join(tempdir, 'eval')

	for d in [saved_model_dir,collect_dir,eval_dir]:
		if os.path.exists(d) is False:
			os.makedirs(d)
	with strategy.scope():
		train_step = train_utils.create_train_step()
		if agent_variety=="sac":
			tf_agent = sac_agent.SacAgent(
			tf_env.time_step_spec(),
			action_spec,
			actor_network=actor_net,
			critic_network=critic_net,
			actor_optimizer=tf.keras.optimizers.Adam(
				learning_rate=actor_learning_rate),
			critic_optimizer=tf.keras.optimizers.Adam(
				learning_rate=critic_learning_rate),
			alpha_optimizer=tf.keras.optimizers.Adam(
				learning_rate=alpha_learning_rate),
			target_update_tau=target_update_tau,
			target_update_period=target_update_period,
			td_errors_loss_fn=tf.math.squared_difference,
			gamma=gamma,
			reward_scale_factor=reward_scale_factor,
			train_step_counter=train_step)

		tf_agent.initialize()
	train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
	]

	rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)
	table_name = 'uniform_table'
	table = reverb.Table(
		table_name,
		max_size=replay_buffer_capacity,
		sampler=reverb.selectors.Uniform(),
		remover=reverb.selectors.Fifo(),
		rate_limiter=reverb.rate_limiters.MinSize(1))

	reverb_server = reverb.Server([table])

	reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
		tf_agent.collect_data_spec,
		sequence_length=2,
		table_name=table_name,
		local_server=reverb_server)

	tf_eval_policy = tf_agent.policy
	eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
	tf_eval_policy, use_tf_function=True)

	tf_collect_policy = tf_agent.collect_policy
	collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)
	random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())

	rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
		reverb_replay.py_client,
		table_name,
		sequence_length=2,
		stride_length=1)

	dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
	experience_dataset_fn = lambda: dataset				

	initial_collect_actor = actor.Actor(
		collect_env,
		random_policy,
		train_step,
		steps_per_run=initial_collect_steps,
		observers=[rb_observer])
	initial_collect_actor.run()

	env_step_metric = py_metrics.EnvironmentSteps()
	collect_actor = actor.Actor(
		collect_env,
		collect_policy,
		train_step,
		steps_per_run=1,
		metrics=actor.collect_metrics(10),
		summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
		observers=[rb_observer, env_step_metric])

	def collect_actor_run_mp(que):
		collect_actor=que.get()
		collect_actor.run()
		que.put(collect_actor)

	eval_actor = actor.Actor(
		eval_env,
		eval_policy,
		train_step,
		episodes_per_run=num_eval_episodes,
		metrics=actor.eval_metrics(num_eval_episodes),
		summary_dir=os.path.join(tempdir, 'eval'),
	)

# Triggers to save the agent's policy checkpoints.
	learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    	triggers.StepPerSecondLogTrigger(train_step, interval=1000),
	]

	agent_learner = learner.Learner(
		tempdir,
		train_step,
		tf_agent,
		experience_dataset_fn,
		triggers=learning_triggers)
	
	def get_eval_metrics():
		eval_actor.run()
		results = {}
		for metric in eval_actor.metrics:
			results[metric.name] = metric.result()
		return results

	metrics = get_eval_metrics()

	def log_eval_metrics(step, metrics):
		eval_results = (', ').join('{} = {:.6f}'.format(name, result) for name, result in metrics.items())
		print('step = {0}: {1}'.format(step, eval_results))

	log_eval_metrics(0, metrics)

	# Reset the train step
	tf_agent.train_step_counter.assign(0)

	metric_history={k:[] for k in metrics.keys()}

	# Evaluate the agent's policy once before training.
	metrics=get_eval_metrics()
	for k,v in metrics.items():
		metric_history[k].append(v)
	avg_return = metrics["AverageReturn"]
	returns = [avg_return]

	que=mp.Queue()
	collect_process=mp.Process(target=collect_actor_run_mp,args=(que,))
	collect_process.start()

	for _ in tqdm(range(num_iterations)):
  		# Training.
		if collect_process.is_alive():
			collect_process.join()
		loss_info = agent_learner.run(iterations=1)
		collect_process=mp.Process(target=collect_actor_run_mp,args=(que,))
		collect_process.start()
  		# Evaluating.
		step = agent_learner.train_step_numpy

		if eval_interval and step % eval_interval == 0:
			metrics = get_eval_metrics()
			log_eval_metrics(step, metrics)
			returns.append(metrics["AverageReturn"])
			for k,v in metrics.items():
				metric_history[k].append(v)

		if log_interval and step % log_interval == 0:
			print('step = {}: loss = {}'.format(step, loss_info.loss.numpy()))

	rb_observer.close()
	reverb_server.stop()
	pd.DataFrame(metric_history).to_csv("metric_history.csv",index=False)
	pd.DataFrame({"returns":returns,"step":[x*eval_interval for x in range(len(returns))]}).to_csv("returns.csv",index=False)