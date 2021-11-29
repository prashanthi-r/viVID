import tensorflow as tf
from config import Config as conf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, PReLU, UpSampling2D, Input, Activation
from tensorflow.keras.models import Sequential, Model
from keras import backend
from keras.applications.vgg19 import VGG19
from keras.losses import mean_squared_error as mse


class Generator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.number_of_residual_blocks = 16

    def build_residual_block(self, residual_block_input):
        """
        This is the residual block
        """
        output = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(residual_block_input)
        output = BatchNormalization(momentum=0.8)(output)
        output = PReLU()(output)
        output = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(output)
        output = BatchNormalization(momentum=0.8)(output)
        return Add()([output, residual_block_input])

    def build_upsampling_block(self, upsampling_input):
        """
        This is the upsampling block
        """
        output = UpSampling2D(size=(2, 2))(upsampling_input)
        output = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(output)
        return PReLU()(output)

    def build_generator(self):
        input_image = Input(shape=self.input_shape)
        # print(f"made it till here -- ! checking for shape of input_image{input_image.shape}")
        pre_residual_block = Sequential(
            [
                Conv2D(64, kernel_size=(9, 9), strides=(1, 1), padding='same'),
                PReLU()
            ]
        )
        low_res_image = pre_residual_block(input_image)
        # print(f"made it till here -- ! checking for shape of low_res_image{low_res_image.shape}")
        residual_block = self.build_residual_block(residual_block_input=low_res_image)
        for r_block in range(0, self.number_of_residual_blocks - 1):
            residual_block = self.build_residual_block(residual_block)
        # print(f"made it till here -- ! checking for shape of residual_block{residual_block.shape}")
        # Post residual blocks
        post_residual_block = Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(momentum=0.8)
        ])
        post_concat_layer = Add()([post_residual_block(residual_block), low_res_image])
        # Upsampling

        print(f"made it till here -- ! checking for shape of post_concat_layer{post_concat_layer.shape}")
        upsampling_output = self.build_upsampling_block(post_concat_layer)
        # Final Layer
        final_layer = Sequential([
            Conv2D(3, kernel_size=(9, 9), strides=(1, 1), padding='same', activation='tanh')
        ])
        return Model(input_image, final_layer(upsampling_output))

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