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
    """
    Contains the loss functions for the generator and discriminator
    """    
    def __init__(self):
        """
        The constructor of the Loss class
        """        
        vgg = VGG19(input_shape=(None, None, 3), include_top=False, weights='imagenet')
        self.vgg1 = Model(inputs=vgg.input, outputs=vgg.layers[9].output)
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()

    def content_loss(self, real, fake):
        """
        Calculates the content loss using vgg19 with imagenet weights

        Args:
            real (tf.tensor): real images for the batch (high resolution images)
            fake (tf.tensor): generated images from the generator model

        Returns:
            [type]: [description]
        """        
        vgg_models = [self.vgg1]
        vgg_loss = 0

        for mod in vgg_models:
            vgg_loss+= K.mean(K.square(mod(fake) - mod(real)))

        return  vgg_loss

    def discriminator_loss(self, real, fake):
        """
        Calculates the discriminator loss of the generator model

        Args:
            real (tf.tensor): real images for the batch (high resolution images)
            fake (tf.tensor): generated images from the generator model

        Returns:
            [type]: [description]
        """        
        return self.bce(tf.ones_like(real), real) + self.bce(tf.zeros_like(fake), fake)

    def perceptual_loss(self, content_loss, fake):
        """
        Calculates the perceptual loss

        Args:
            content_loss (tf.tensor): content loss calculated using vgg19 
            generator_loss (tf.tensor): binary cross entropy of the generator model

        Returns:
            perceptual loss (tf.tensor): the sum of the content loss and the binary cross entropy of the generator model * 1e-2
        """        
        return content_loss + 1e-3 * self.bce(tf.ones_like(fake), fake)
