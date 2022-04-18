import ray
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

class BasicModel(TFModelV2):
	"""Example of a keras custom model that just delegates to an fc-net."""
	def __init__(self, obs_space, action_space, num_outputs, model_config, name):
		super(BasicModel, self).__init__(
			obs_space, action_space, num_outputs, model_config, name
		)
		self.model = FullyConnectedNetwork(
			obs_space, action_space, num_outputs, model_config, name
		)

	def forward(self, input_dict, state, seq_lens):
		model_out, model_state=self.model.forward(input_dict, state, seq_lens)
		model_out=tf.cast(model_out,tf.float32)
		return model_out,model_state

	def value_function(self):
		return self.model.value_function()