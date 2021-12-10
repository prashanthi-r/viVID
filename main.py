from discriminator import Discriminator
from generator import Generator
from tensorflow.keras import Input
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
from tensorflow.keras.layers import Input
import logging
from tensorflow.keras.losses import MeanSquaredError
from loss import Loss
import os
import datetime
from tensorflow.keras.applications.vgg19 import preprocess_input
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SRGAN')


class SRGAN:

    def __init__(self):
        self.dataset = "DIV2K"
        self.pretrain_generator = conf.pretrain_generator
        self.starting_epoch = conf.starting_epoch
        self.pretrain_number_of_epochs = conf.pretrain_generator_epochs
        self.dataset_directory = conf.data_dir
        self.scalling_factor = conf.scale
        self.low_resolution_images, self.high_resolution_images = self.preprocess()
        self.input_shape = conf.input_shape
        self.input_shape_lr = conf.input_shape_lr
        self.batch_size = conf.batch_size
        self.optimizer = conf.optimizer
        self.generator_optimizer = conf.optimizer
        self.discriminator_optimizer = conf.optimizer
        generator = Generator(input_shape=self.input_shape_lr)
        self.generator = generator.build_generator()
        if self.pretrain_generator:
            self.generator.load_weights(conf.generator_weights_path)
        discriminator = Discriminator(self.input_shape)
        self.discriminator = discriminator.build_discriminator()
        self.loss = Loss()
        self.mse = MeanSquaredError()
        self.epochs = conf.epochs
        self.loss_file_path = 'loss_logs.json'

    def preprocess(self):
        """
        The images are preprocessed to be used in the model. 

        Returns:
            np.array : The low resolution images and high resolution images
        """
        return get_data_clip(self.dataset_directory, self.scalling_factor, patch_size=96, seed=7, patches_count=8,
                             gray_scale=False)

    def show_images(self, images, fig_name, res, epoch, loc):
        """
        This function shows the images in a grid

        Args:
            images (tf.tensor): Array of images
            fig_name (str): Name of the figure
            res (str): Type of image (low or high)
            epoch (int): Epoch number
            loc (str): path to save the figure
        """
        if res == 'high':
            # -1 to 1
            images = ((np.array(images) + 1) * 127.5).astype(np.uint8)
        else:
            # 0 to 1
            images = (np.array(images) * 255).astype(np.uint8)

        images = np.array(images)
        print(
            f"fig {fig_name}\nimg min:{min(np.reshape(images[0], [-1]))} max:{max(np.reshape(images[0], [-1]))}")
        fig = plt.figure(figsize=(1, images.shape[0]))
        print(
            f"fig {fig_name}\nimg min:{min(np.reshape(images[0], [-1]))} max:{max(np.reshape(images[0], [-1]))}")
        gs = gridspec.GridSpec(1, images.shape[0])
        gs.update(wspace=0.05, hspace=0.05)
        fig.set_size_inches(7.5, 7.5)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img)
            plt.title(fig_name)
        plt.savefig(loc + '/' + fig_name + str(epoch) + '.png')
        return

    def train(self):
        """
        This function trains the SRGAN model
        """
        if self.pretrain_generator:
            for epoch in range(self.pretrain_number_of_epochs):
                for batch in range(0, len(self.high_resolution_images), self.batch_size):
                    low_res_batch = self.low_resolution_images[batch:batch +
                                                               self.batch_size]
                    high_res_batch = self.high_resolution_images[batch:batch + self.batch_size]
                    with tf.GradientTape() as tape:
                        generated_samples = self.generator(
                            low_res_batch, training=True)
                        mse_loss = self.mse(generated_samples, high_res_batch)
                    logger.info(
                        f"{'---' * 30} EPOCH : {epoch} LOSS: {mse_loss} {'---' * 30}")
                    grad = tape.gradient(
                        mse_loss, self.generator.trainable_weights)
                    self.generator_optimizer.apply_gradients(
                        zip(grad, self.generator.trainable_weights))
        today = datetime.datetime.now()
        date_time = today.strftime("%m_%d_%Y_%H_%M_%S")
        folder_loc = 'output' + date_time
        loss_data = []
        if not os.path.exists(folder_loc):
            os.makedirs(folder_loc)
        logger.info(f"{'---' * 30} STARTING TRAINING {'---' * 30}")
        for epoch in range(0, self.epochs):
            logger.info(
                f"{'---' * 30} EPOCH : {epoch} CONTINUING EPOCH: {epoch + self.starting_epoch} {'---' * 30}")
            number_of_batches = 0
            for batch in range(0, len(self.high_resolution_images), self.batch_size):
                low_res_batch = self.low_resolution_images[batch:batch +
                                                           self.batch_size]
                high_res_batch = self.high_resolution_images[batch:batch + self.batch_size]
                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    generated_samples = self.generator(
                        low_res_batch, training=True)
                    logits_real = self.discriminator(
                        high_res_batch, training=True)
                    logits_fake = self.discriminator(
                        generated_samples, training=True)
                    content_loss = self.loss.content_loss(
                        real=high_res_batch, fake=generated_samples)
                    perceptual_loss = self.loss.perceptual_loss(content_loss=content_loss,
                                                                fake=logits_fake)
                    discriminator_loss = self.loss.discriminator_loss(
                        real=logits_real, fake=logits_fake)
                    number_of_batches += 1
                    logger.info(
                        f"BATCH: {number_of_batches} | PERCEPUTAL LOSS: {perceptual_loss} | DISCRIMINATOR LOSS: {discriminator_loss}")
                generator_gradients = generator_tape.gradient(
                    perceptual_loss, self.generator.trainable_variables)
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                                      self.discriminator.trainable_variables)
                self.generator_optimizer.apply_gradients(
                    zip(generator_gradients, self.generator.trainable_variables))
                self.discriminator_optimizer.apply_gradients(
                    zip(discriminator_gradients, self.discriminator.trainable_variables))
                loss_data.append({
                    "epoch": epoch + self.starting_epoch,
                    "perceptual_loss": perceptual_loss.numpy().astype('str'),
                    "content_loss": content_loss.numpy().astype('str'),
                    "discriminator_loss": discriminator_loss.numpy().astype('str')
                })
                if epoch % 20 == 0:
                    self.save_model_weights(self.starting_epoch + epoch)
                    self.save_logs(loss_data)
                if epoch % 10 == 0:
                    if conf.GCP:
                        self.show_images(
                            high_res_batch[:3], 'input_highres_img', 'high', epoch, folder_loc)
                        self.show_images(
                            low_res_batch[:3], 'input_lowres_img', 'low', epoch, folder_loc)
                        self.show_images(
                            generated_samples[:3], 'generated_img', 'high', epoch, folder_loc)
                    else:
                        self.show_images(
                            high_res_batch[:3], 'input_highres_img', 'high', epoch, folder_loc)
                        self.show_images(
                            low_res_batch[:3], 'input_lowres_img', 'low', epoch, folder_loc)
                        self.show_images(
                            generated_samples[:3], 'generated_img', 'high', epoch, folder_loc)

    def save_model_weights(self, epoch):
        """
        This function saves the model weights for a given epoch

        Args:
            epoch (int): epoch number
        """
        output_dir = os.path.join("model_ckpts", "srgan")
        output_path = os.path.join(output_dir, "srgan")
        os.makedirs("model_ckpts", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self.generator.save(output_path + '/gen' + str(epoch) + '.h5')
        logger.info("SRGAN Model is saved")

    def save_logs(self, data):
        """
        This function saves the loss data for a given number of epochs

        Args:
            data ([type]): [description]
        """
        with open('logs.json', 'w+') as f:
            json.dump(data, f)
        f.close()


if __name__ == "__main__":
    # Create the SRGAN model
    srgan = SRGAN()
    # Train the SRGAN model
    srgan.train()