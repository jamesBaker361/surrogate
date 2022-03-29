from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.keras_layers import inner_reshape
from tf_agents.metrics import tf_metrics
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


dense = functools.partial(
	tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(
        scale=1./ 3.0, mode='fan_in', distribution='uniform'))


def create_identity_layer():
    return tf.keras.layers.Lambda(lambda x: x)


def create_fc_network(layer_units):
    return sequential.Sequential([dense(num_units) for num_units in layer_units])


def create_actor_network(fc_layer_units, action_spec):
	"""Create an actor network for DDPG."""
	flat_action_spec = tf.nest.flatten(action_spec)
	if len(flat_action_spec) > 1:
		raise ValueError('Only a single action tensor is supported by this network')
	flat_action_spec = flat_action_spec[0]

	fc_layers = [dense(num_units) for num_units in fc_layer_units]

	num_actions = flat_action_spec.shape.num_elements()
	action_fc_layer = tf.keras.layers.Dense(
      num_actions,
      activation=tf.keras.activations.sigmoid,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003))
	scaling_layer = tf.keras.layers.Lambda(
      lambda x: common.scale_to_spec(x, flat_action_spec))
	return sequential.Sequential(fc_layers + [action_fc_layer, scaling_layer])

def create_critic_network(obs_fc_layer_units,
                          action_fc_layer_units,
                          joint_fc_layer_units):
	"""Create a critic network for DDPG."""

	def split_inputs(inputs):
		return {'observation': inputs[0], 'action': inputs[1]}

	obs_network = create_fc_network(
    	obs_fc_layer_units) if obs_fc_layer_units else create_identity_layer()
	action_network = create_fc_network(
    	action_fc_layer_units
	) if action_fc_layer_units else create_identity_layer()
	joint_network = create_fc_network(
      joint_fc_layer_units) if joint_fc_layer_units else create_identity_layer()
	value_fc_layer = tf.keras.layers.Dense(
      1,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003))

	return sequential.Sequential([
      tf.keras.layers.Lambda(split_inputs),
      nest_map.NestMap({
          'observation': obs_network,
          'action': action_network
      }),
      nest_map.NestFlatten(),
      tf.keras.layers.Concatenate(),
      joint_network,
      value_fc_layer,
      inner_reshape.InnerReshape([1], [])
  ])