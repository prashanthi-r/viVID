import numpy as np
import os
import PIL
from tensorflow.keras import layers, Sequential
import tensorflow as tf
import time

class Discriminator:

	def __init__(self,input_shape):
		self.input_shape = input_shape

	def build_discriminator(self):
		model = Sequential()
		model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
										input_shape=self.input_shape))
		model.add(layers.LeakyReLU())
		
		# Block 1
		model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
		model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
		model.add(layers.LeakyReLU())
		
		# Block 2
		model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
		model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
		model.add(layers.LeakyReLU())

		# Block 3
		model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
		model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
		model.add(layers.LeakyReLU())

		# Block 4
		model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
		model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
		model.add(layers.LeakyReLU())

		# Block 5
		model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
		model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
		model.add(layers.LeakyReLU())

		# Block 6
		model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
		model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
		model.add(layers.LeakyReLU())

		# Block 7
		model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
		model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
		model.add(layers.LeakyReLU())


		model.add(layers.Flatten())

		model.add(layers.Dense(1024))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1,activation='sigmoid'))

		return model

	def loss_function(self,logits_fake: tf.Tensor, logits_real: tf.Tensor) -> tf.Tensor:
		D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
		D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
		return D_loss