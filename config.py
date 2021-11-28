import tensorflow as tf

class Config:
	data_dir = '/Users/rugvedmavidipalli/Downloads/archive/pokemon_jpg'
	epochs = 5
	batch_size = 16
	learning_rate = 1e-3
	beta = 0.9
	rescaling_factor = 0.006
	num_channels = 3
	input_shape = (batch_size,256,256,num_channels)
	optimizer = tf.keras.optimizers.Adam(learning_rate)