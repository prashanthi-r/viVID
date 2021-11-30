import tensorflow as tf
from config import Config as conf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, PReLU, UpSampling2D, Input, Activation, Lambda
from tensorflow.keras.models import Sequential, Model
from keras import backend
from keras.applications.vgg19 import VGG19
from keras.losses import mean_squared_error as mse


class Generator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.number_of_residual_blocks = 16

    def SubpixelConv2D(self, scale):
        return Lambda(lambda x: tf.nn.depth_to_space(x, scale))

    def build_residual_block(self, residual_block_input):
        """
        This is the residual block
        """
        output = Conv2D(64, kernel_size=3, strides=1, padding='same')(residual_block_input)
        output = BatchNormalization(momentum=0.8)(output)
        output = PReLU()(output)
        output = Conv2D(64, kernel_size=3, strides=1, padding='same')(output)
        output = BatchNormalization(momentum=0.8)(output)
        return Add()([output, residual_block_input])

    def build_upsampling_block(self, upsampling_input):
        """
        This is the upsampling block
        """
        output = Conv2D(filters=256, kernel_size=3, padding='same')(upsampling_input)
        output = self.SubpixelConv2D(2)(output)
        return PReLU()(output)

    def build_generator(self):
        input_image = Input(shape=self.input_shape)
        pre_residual_block = Conv2D(64, kernel_size=9, strides=1, padding='same')(input_image)
        pre_residual_block = PReLU()(pre_residual_block)
        residual_block = self.build_residual_block(residual_block_input=pre_residual_block)
        for r_block in range(0, self.number_of_residual_blocks - 1):
            residual_block = self.build_residual_block(residual_block)
        post_residual = Conv2D(64, kernel_size=3, strides=1, padding='same')(residual_block)
        post_residual = BatchNormalization(momentum=0.8)(post_residual)
        post_residual = Add()([post_residual, pre_residual_block])
        upsample_layer = self.build_upsampling_block(upsampling_input=post_residual)
        for upsample_block in range(int(2 - 1)):
            upsample_layer = self.build_upsampling_block(upsample_layer)
        final_output = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(upsample_layer)
        return Model(input_image, final_output)

    def loss_function(self, fake_imgs: tf.Tensor, real_imgs: tf.Tensor, logits_fake: tf.Tensor, logits_real: tf.Tensor,
                      i=5, j=4) -> tf.Tensor:
        model = VGG19(weights='imagenet')
        model = backend.function([model.layers[0].input], [model.layers[i * j].output])
        vgg_input_features = model(real_imgs)[0]
        vgg_target_features = model(fake_imgs)[0]

        content_loss = mse(vgg_input_features, vgg_target_features)
        adversarial_loss = 1e-3 * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
        feature_dim = vgg_input_features.shape[1] * vgg_input_features.shape[2]
        perceptual_loss = conf.rescaling_factor * (1 / feature_dim) * content_loss + adversarial_loss

        return perceptual_loss
