import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, PReLU, UpSampling2D, Input, Activation, Lambda
from tensorflow.keras.models import  Model
from tensorflow.keras.losses import mean_squared_error as mse


class Generator:
    def __init__(self, input_shape):
        """
        The constructor of the Generator class

        Args:
            input_shape (tuple): (height, width, channels)
        """        
        self.input_shape = input_shape
        self.number_of_residual_blocks = 16
        self.number_of_upsampling_blocks = 2

    def SubpixelConv2D(self, scale):
        """
        This is the upsampling layer

        Args:
            scale (int): The scale of the upsampling to be applied to the image

        Returns:
            tf.tensor: scaled image output
        """        
        return Lambda(lambda x: tf.nn.depth_to_space(x, scale))

    def build_residual_block(self, residual_block_input):
        """
        This method builds a residual block for the generator model
        
        Architecture:
            - Conv2D(64, kernel_size=3, strides=1, padding='same')
            - BatchNormalization(momentum=0.8)
            - PReLU()
            - BatchNormalization(momentum=0.8)
            - Add()
        Args:
            residual_block_input ([type]): [description]

        Returns:
            tf.tensor: residual block output
        """        
        output = Conv2D(64, kernel_size=3, strides=1, padding='same')(residual_block_input)
        output = BatchNormalization(momentum=0.8)(output)
        output = PReLU()(output)
        output = Conv2D(64, kernel_size=3, strides=1, padding='same')(output)
        output = BatchNormalization(momentum=0.8)(output)
        return Add()([output, residual_block_input])

    def build_upsampling_block(self, upsampling_input):
        """
        This is builds the upsampling block for the generator model
        
        Architecture:
             - Conv2D(96, kernel_size=3, strides=1, padding='same')
             - SubpixelConv2D(scale=2)
             - PReLU()

        Args:
            upsampling_input (tf.tensor): Output of the upsampling layer

        Returns:
            tf.tensor: upsampling block output
        """        
        output = Conv2D(filters=96, kernel_size=3, padding='same')(upsampling_input)
        output = self.SubpixelConv2D(2)(output)
        return PReLU()(output)

    def build_generator(self):
        """
        This builds the generator model

        Returns:
            keras.model: returns the generator model
        """        
        input_image = Input(shape=self.input_shape)
        pre_residual_block = Conv2D(64, kernel_size=9, strides=1, padding='same')(input_image)
        pre_residual_block = PReLU()(pre_residual_block)
        residual_block = self.build_residual_block(residual_block_input=pre_residual_block)
        for redsi_block in range(0, self.number_of_residual_blocks - 1):
            residual_block = self.build_residual_block(residual_block)
        post_residual = Conv2D(64, kernel_size=3, strides=1, padding='same')(residual_block)
        post_residual = BatchNormalization(momentum=0.8)(post_residual)
        post_residual = Add()([post_residual, pre_residual_block])
        upsample_layer = self.build_upsampling_block(upsampling_input=post_residual)
        for upsample_block in range(self.number_of_upsampling_blocks - 1):
            upsample_layer = self.build_upsampling_block(upsample_layer)
        final_output = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(upsample_layer)
        return Model(input_image, final_output)

