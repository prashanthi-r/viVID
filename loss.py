from keras import backend
from keras.applications.vgg19 import VGG19
from keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
from tensorflow import keras
from config import Config as conf


class Loss:
    def __init__(self):
        vgg = VGG19(input_shape=(None, None, 3), include_top=False)
        self.vgg = Model(inputs=vgg.input, outputs=vgg.layers[20].output)
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()
        self.learning_rate = conf.learning_rate

    def content_loss(self, real, fake):
        """
        Calculates the content loss using vgg19 with imagenet weights
        """
        return self.mse((self.vgg(preprocess_input(real)) / 12.75), (self.vgg(preprocess_input(fake)) / 12.75))

    def generator_loss(self, fake):
        """
        Calculates the binary cross entropy of the generator model
        """
        return self.bce(tf.ones_like(fake), fake)

    def discriminator_loss(self, real, fake):
        """
        Calculates the discriminator loss of the generator model
        """
        return self.bce(tf.ones_like(real), real) + self.bce(tf.zeros_like(fake), fake)

    def perceptual_loss(self, content_loss, generator_loss):
        """
        Calculates the perceptual loss
        """
        return content_loss + self.learning_rate * generator_loss
