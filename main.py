from config import Config as conf
from preprocessing import get_data_clip,test_preprocessing, test_video
from discriminator import Discriminator
from generator import Generator
from tensorflow.keras import Input
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Input
import logging
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from loss import Loss
import os
import datetime
from tensorflow.keras.applications.vgg19 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SRGAN')


class SRGAN:

    def __init__(self):
        self.dataset = ""
        self.dataset_directory = conf.data_dir
        self.scalling_factor = conf.scale
        #self.low_resolution_images, self.high_resolution_images = self.preprocess()
        self.input_shape = conf.input_shape
        self.input_shape_lr = conf.input_shape_lr
        self.batch_size = conf.batch_size
        self.learning_rate = conf.learning_rate
        self.optimizer = conf.optimizer
        self.generator_optimizer = conf.optimizer
        self.discriminator_optimizer = conf.optimizer
        generator = Generator(input_shape=self.input_shape_lr)
        self.generator = generator.build_generator()
        discriminator = Discriminator(self.input_shape)
        self.discriminator = discriminator.build_discriminator()
        self.loss = Loss()
        self.mse = MeanSquaredError()
        self.epochs = conf.epochs

    def preprocess(self):
        return get_data_clip(self.dataset_directory, self.scalling_factor, patch_size =96, seed=5, patches_count=2, gray_scale = False)

    def show_images(self, images, fig_name , res, epoch, loc):
        if res == 'high':
            #-1 to 1
            images = ((np.array(images) + 1) * 127.5).astype(np.uint8)
        elif res == 'low':
            # 0 to 1
            images = (np.array(images)*255).astype(np.uint8)
    

        images =  np.array(images)
        print(f"fig {fig_name}\nimg min:{min(np.reshape(images[0],[-1]))} max:{max(np.reshape(images[0],[-1]))}")
        fig = plt.figure(figsize=(1, images.shape[0]))
        print(f"fig {fig_name}\nimg min:{min(np.reshape(images[0],[-1]))} max:{max(np.reshape(images[0],[-1]))}")
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
        plt.savefig(loc+'/'+fig_name+str(epoch)+'.png')
        return

    def train(self):
        '''n_step_epoch = 100
        for epoch in range(n_step_epoch):
            print('Epoch: ',epoch)
            for batch in range(0, len(self.high_resolution_images), self.batch_size):
                low_res_batch = self.low_resolution_images[batch:batch + self.batch_size]
                high_res_batch = self.high_resolution_images[batch:batch + self.batch_size]
                with tf.GradientTape() as tape:
                    generated_samples = self.generator(low_res_batch, training=True)
                    #high_res_batch = ((np.array(high_res_batch) + 1) * 127.5).astype(np.uint8)
                    #generated_samples = ((np.array(generated_samples) + 1) * 127.5).astype(np.uint8)
                    mse_loss = self.mse(generated_samples, high_res_batch)
                grad = tape.gradient(mse_loss, self.generator.trainable_weights)
                self.generator_optimizer.apply_gradients(zip(grad, self.generator.trainable_weights))'''
        import os
        
        today = datetime.datetime.now()
        date_time = today.strftime("%m_%d_%Y_%H_%M_%S")
        folder_loc = 'output'+ date_time

        if not os.path.exists(folder_loc):
            os.makedirs(folder_loc)

        logger.info(f"{'---' * 30} STARTING TRAINING {'---' * 30}")
        for epoch in range(0, self.epochs):
            logger.info(f"{'---' * 30} EPOCH : {epoch} {'---' * 30}")
            number_of_batches = 0
            for batch in range(0, len(self.high_resolution_images), self.batch_size):
                low_res_batch = self.low_resolution_images[batch:batch + self.batch_size]
                high_res_batch = self.high_resolution_images[batch:batch + self.batch_size]

                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    generated_samples = self.generator(low_res_batch, training=True)
                    logits_real = self.discriminator(high_res_batch, training=True)
                    logits_fake = self.discriminator(generated_samples, training=True)
                    content_loss = self.loss.content_loss(real=high_res_batch, fake=generated_samples)
                    generator_loss = self.loss.generator_loss(fake=logits_fake)
                    perceptual_loss = self.loss.perceptual_loss(content_loss=content_loss,
                                                                generator_loss=generator_loss)
                    discriminator_loss = self.loss.discriminator_loss(real=logits_real, fake=logits_fake)
                    number_of_batches += 1
                    logger.info(
                        f"BATCH: {number_of_batches} | PERCEPUTAL LOSS: {perceptual_loss} | DISCRIMINATOR LOSS: {discriminator_loss}")
                generator_gradients = generator_tape.gradient(perceptual_loss, self.generator.trainable_variables)
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                                      self.discriminator.trainable_variables)
            
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
                self.discriminator_optimizer.apply_gradients(
                    zip(discriminator_gradients, self.discriminator.trainable_variables))

            if epoch%50==0:
                self.save_model_weights(epoch)
            
            if epoch%10==0:
                if conf.GCP:
                    self.show_images(high_res_batch[:3], 'input highres img', 'high', epoch, folder_loc)
                    self.show_images(low_res_batch[:3], 'input lowres img', 'low', epoch, folder_loc)
                    self.show_images(generated_samples[:3], 'generated img', 'high', epoch, folder_loc)
                else:
                    self.show_images(high_res_batch[:3], 'input highres img', 'high', epoch, folder_loc)
                    plt.show()
                    self.show_images(low_res_batch[:3], 'input lowres img', 'low', epoch, folder_loc)
                    plt.show()
                    self.show_images(generated_samples[:3], 'generated img', 'high', epoch, folder_loc) 
                    plt.show()
                

        return perceptual_loss, discriminator_loss

    def save_model_weights(self,epoch):
        output_dir = os.path.join("model_ckpts", "srgan")
        output_path = os.path.join(output_dir, "srgan")
        os.makedirs("model_ckpts", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self.generator.save(output_path+'/gen'+str(epoch)+'.h5')
        logger.info("SRGAN Model is saved")

    def generate_output(self):
        print(self.low_resolution_images.shape)
        print(self.high_resolution_images.shape)
        model=self.generator
        model.load_weights(tf.train.latest_checkpoint("/Users/adidot/Desktop/Trial SRGAN/viVID/srgan-13"))
        self.show_images(self.low_resolution_images,'input lowres img','low')
        plt.show()
        generated_ouput = model(self.low_resolution_images)
        self.show_images(generated_ouput,'generated img','high')
        plt.show()

    def pred_pipeline(self):

        generator = Generator(input_shape=self.input_shape_lr)  
        generator = generator.build_generator()      
        generator.load_weights("gen1740.h5")
        recon_iamge, lr_recon_iamge = test_preprocessing(generator, '../test_images/', conf.scale, conf.image_height)        
        print('recon_iamge shape',recon_iamge.shape)
        self.show_images(recon_iamge,'Output High Resolution Image','F',16,'../output/')
        plt.show()
        self.show_images(lr_recon_iamge,'Input Low Resolution Image','low',16,'../output/')
        plt.show()
        
        # print(f"lr_recon_iamge:{lr_recon_iamge.shape}\n Output_result:{recon_iamge.shape}")
        # generated_ouput = model(self.low_resolution_images)
        # self.show_images(generated_ouput,'generated img','high')
        # plt.show()
    def pred_pipeline_video(self):

        generator = Generator(input_shape=self.input_shape_lr)  
        generator = generator.build_generator()      
        generator.load_weights("gen1740.h5")
        recon_iamge, lr_recon_iamge = test_video(generator, '../test_images/', conf.scale, conf.image_height)        
        # print('recon_iamge shape',recon_iamge.shape)
        # self.show_images(recon_iamge,'Output High Resolution Image','F',16,'../output/')
        # plt.show()
        # self.show_images(lr_recon_iamge,'Input Low Resolution Image','low',16,'../output/')
        # plt.show()

# def test():

# def eval_model():


if __name__ == "__main__":
    srgan = SRGAN()
    # srgan.pred_pipeline()
    srgan.pred_pipeline_video()
    #srgan.generate_output()
    #perceptual_loss, discriminator_loss = srgan.train()
    

# Visualization Snippet
# if epoch%10==0:
#     self.show_images(high_res_batch[:3],'input highres img','high')
#     plt.show()
#     self.show_images(low_res_batch[:3],'input lowres img','low')
#     plt.show()
#     self.show_images(generated_samples[:3],'generated img','high')
#     plt.show()
#     self.save_model_weights(epoch)
