from config import Config as conf
from preprocessing import get_data
from discriminator import Discriminator
from generator import Generator
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, PReLU, UpSampling2D, Input, Activation, Lambda, \
    LeakyReLU, Dense, Reshape
from keras import backend
from tensorflow.keras.models import Sequential, Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from preprocessing import get_data
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('SRGANMODEL')


class SRGANMODEL:
    def __init__(self):
        self.low_resolution_shape = conf.input_shape_lr
        self.high_resolution_shape = conf.input_shape_hr
        self.learning_rate_generator = 0.0001
        self.generator = None
        self.discriminator = None
        self.vgg = None
        self.srgan = None
        self.build_and_compile_vgg()
        self.build_and_compile_generator()
        self.build_and_compile_discriminator()
        self.build_and_compile_full_srgan_model()
        logger.info("Preprocessing Data")
        self.low_resolution_images, self.high_resolution_images = get_data(data_dir=conf.data_dir, scale=conf.scale)
        self.number_of_epochs = conf.epochs
        self.batch_size = conf.batch_size

    def build_and_compile_generator(self):
        generator = Generator(self.low_resolution_shape)
        self.generator = generator.build_generator()
        self.generator.compile(
            loss=['mse'],
            optimizer=tf.keras.optimizers.Adam(self.learning_rate_generator, 0.9),
            metrics=['mse']
        )

    def build_and_compile_discriminator(self):
        # TODO Change back to our implimentation using this just to test rest of the code
        # TODO Change back to our implimentation the bellow code will essentially work if we use our implimentation
        # discriminator = Discriminator(self.high_resolution_shape)
        # self.discriminator = discriminator.build_discriminator()
        # self.discriminator.compile(
        #     loss=['binary_crossentropy'],
        #     optimizer=tf.keras.optimizers.Adam(self.learning_rate_generator, 0.9),
        #     metrics=['accuracy']
        # )

    def build_and_compile_vgg(self):
        vgg = VGG19(weights="imagenet")
        self.vgg = Model(inputs=vgg.input, outputs=vgg.layers[20].output)
        img = Input(shape=self.high_resolution_shape)
        self.vgg = Model(img, self.vgg(img))
        self.vgg.trainable = False
        self.vgg.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(0.0001, 0.9),
            metrics=['accuracy']
        )

    def build_and_compile_full_srgan_model(self):
        generated_image = self.generator(Input(self.low_resolution_shape))
        features = self.vgg(
            generated_image
        ) # Breaks Here
        self.discriminator.trainable = False
        self.srgan = Model([Input(shape=self.low_resolution_shape), Input(shape=self.high_resolution_shape)],
                           [self.discriminator(generated_image), features])
        self.srgan.compile(
            loss=['binary_crossentropy', 'mse'],
            loss_weights=[1e-3, 1],
            optimizer=tf.keras.optimizers.Adam(self.learning_rate_generator, 0.9)
        )

    def train(self):
        logger.info("Starting Training of SRGAN MODEL")
        # TODO why? change this
        # Source https://github.com/MathiasGruber/SRGAN-Keras/blob/c83cbbe1bc3c5d6b8acc929399218c609ae32035/libs/srgan.py#L182
        disciminator_output_shape = list(self.discriminator.output_shape)
        disciminator_output_shape[0] = self.batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)
        real = np.ones(disciminator_output_shape)
        fake = np.zeros(disciminator_output_shape)
        for epoch in range(self.number_of_epochs):
            logger.info(f"Training On Epoch {epoch}")
            logger.info("-" * 30)
            for batch in range(0, len(self.high_resolution_images), conf.batch_size):
                low_resolution_batch = self.low_resolution_images[batch:batch + self.batch_size]
                high_resolution_batch = self.high_resolution_images[batch:batch + self.batch_size]

                generated_high_resolution_images = self.generator.predict(low_resolution_batch)
                real_loss = self.discriminator.train_on_batch(high_resolution_batch, real)
                fake_loss = self.discriminator.train_on_batch(generated_high_resolution_images, fake)
                # TODO change loss to our loss implimentation
                discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
                high_resolution_features = self.vgg.predict(high_resolution_batch)
                generator_loss = self.srgan.train_on_batch(low_resolution_batch, [real, high_resolution_features])
                logger.info(f"Training on Batch Complete: {batch}")
                logger.info(f"Generator Loss: {generator_loss}, Discriminator Loss: {discriminator_loss}")
                logger.info("-" * 30)
        logger.info("Completed Training Sucessfully")


if __name__ == '__main__':
    srgan = SRGANMODEL()
