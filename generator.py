import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, PReLU, UpSampling2D
from tensorflow.keras.models import Sequential, Model


class Generator:
	def __init__(self,input_shape):
		self.input_shape = input_shape
		self.number_of_residual_blocks = 16
		self.number_of_upsampling_blocks = 2

	def build_residual_block(self, residual_block_input):
		"""
		This is the residual block
		"""
		residual_block = Sequential([
			Conv2D(filters=64, kernel_size=3, strides=1,
				   padding='same'),
			BatchNormalization(momentum=0.8),
			PReLU(),
			Conv2D(filters=64, kernel_size=3, strides=1,
				   padding='same'),
			Add([BatchNormalization(momentum=0.8), residual_block_input])
		])
		# concat_layer = Add()[residual_block(residual_block_input), residual_block_input]
		return residual_block

	def build_upsampling_block(self):
		"""
		This is the upsampling block
		"""
		upsampling_model = Sequential([
			UpSampling2D(size=2),
			Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')
		])
		return upsampling_model

	def build_generator(self):
		input_image = Input(shape=self.input_shape)
		pre_residual_block = Sequential(
			[
				Conv2D(64, kernel_size=9, strides=1, padding='same'),
				PReLU()
			]
		)
		low_res_image = pre_residual_block(input_image)
		residual_block = self.build_residual_block(residual_block_input=low_res_image)
		for r_block in range(0, self.number_of_residual_blocks - 1):
			residual_block = residual_block(residual_block)
		# Post residual blocks
		post_residual_block = Sequential([
			Conv2D(64, kernel_size=3, strides=1, padding='same'),
			BatchNormalization(momentum=0.8)
		])
		post_concat_layer = Add()([post_residual_block(residual_block), pre_residual_block(input_image)])
		# Upsampling
		upsampling = self.build_upsampling_block()
		upsampling_output = upsampling(post_concat_layer)
		upsampling_output = upsampling(upsampling_output)
		# Final Layer
		final_layer = Sequential([
			Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')
		])
		return Model(input_image, final_layer(upsampling_output))

	def loss_function(self,fake_imgs: tf.Tensor, real_imgs: tf.Tensor,logits_fake: tf.Tensor, logits_real: tf.Tensor,i=5,j=4) -> tf.Tensor:
		model = VGG19(weights='imagenet')
		model = backend.function([model.layers[0].input], [model.layers[i*j].output])
		vgg_input_features = model(real_imgs)[0]
		vgg_target_features = model(fake_imgs)[0]

		content_loss = mse(vgg_input_features,vgg_target_features)
		adversarial_loss = 1e-3 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
		feature_dim = vgg_input_features.shape[1]*vgg_input_features.shape[2]
		perceptual_loss = (1/feature_dim)*content_loss + adversarial_loss

		return perceptual_loss