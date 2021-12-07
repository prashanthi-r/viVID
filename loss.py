from keras import backend
from keras.applications.vgg19 import VGG19
from keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
import numpy as np
from config import Config as conf


class Loss:
    def __init__(self):
        vgg = VGG19(input_shape=(None, None, 3), include_top=False, weights='imagenet')
        self.vgg1 = Model(inputs=vgg.input, outputs=vgg.layers[5].output)
        self.vgg2 = Model(inputs=vgg.input, outputs=vgg.layers[10].output)
        self.vgg3 = Model(inputs=vgg.input, outputs=vgg.layers[15].output)
        #self.vgg4 = Model(inputs=vgg.input, outputs=vgg.layers[20].output)
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()
        self.learning_rate = conf.learning_rate

    def content_loss(self, real, fake):
        """
        Calculates the content loss using vgg19 with imagenet weights
        """

        self.vgg1.trainable = False
        self.vgg2.trainable = False
        self.vgg3.trainable = False
        vgg_models = [self.vgg1,self.vgg2,self.vgg3]
        vgg_loss = 0

        for mod in vgg_models:
            vgg_loss+= K.mean(K.square(mod(fake) - mod(real)))

        return  vgg_loss/3

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




# TEST CODE
#real = ((np.array(real) + 1) * 127.5).astype(np.uint8)
#fake = (np.array(fake)*255).astype(np.uint8)
#print(self.mse((self.vgg(real)), (self.vgg(fake))))
#return self.mse((self.vgg(real))/12.75, (self.vgg(fake))/12,.75)