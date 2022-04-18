import tensorflow as tf
import numpy as np

def get_datasets(batch_size=32):
	data=np.load("samples.npz")

	x=data["x"]
	y=data["y"]

	x_train=x[:int(.8*len(x))]
	x_test=x[int(.8*len(x)):]

	y_train=y[:int(.8*len(y))]
	y_test=y[int(.8*len(y)):]

	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	# Shuffle and slice the dataset.
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

	# Now we get a test dataset.
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	test_dataset = test_dataset.batch(batch_size)

	return train_dataset,test_dataset

def add_dense_layer(nodes,model):
	model.add([
		tf.keras.layers.Dense(nodes),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.LeakyReLU()
		])

def get_model(input_shape):
	model=tf.keras.Sequential([
		tf.keras.layers.Dense(512, input_shape=input_shape)
	])
	for n in [256,128,128,256,512]:
		model.add_dense_layer(n)