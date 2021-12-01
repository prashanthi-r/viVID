from config import Config as conf
from preprocessing import get_data
from discriminator import Discriminator
from generator import Generator
from tensorflow.keras import Input
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input

class SRGAN:

    def __init__(self):
        self.input_shape = conf.input_shape
        self.input_shape_lr = conf.input_shape_lr
        self.batch_size = conf.batch_size
        self.learning_rate = conf.learning_rate
        self.optimizer = conf.optimizer

    def preprocess(self):
        return get_data(conf.data_dir, conf.scale)

    def train(self, LR_image, HR_image):
        discriminator = Discriminator(self.input_shape)
        generator = Generator(input_shape = self.input_shape_lr, model = VGG19(weights='imagenet',include_top=False,
	input_tensor=Input(shape=(256, 256, 3))), phi_i=4,phi_j=5) # Does not support batches

        d_model = discriminator.build_discriminator()
        # TODO change the loss
        g_model = generator.build_generator()
        for i in range(0, len(HR_image), self.batch_size):
            LR_batch = LR_image[i:i + self.batch_size]
            HR_batch = HR_image[i:i + self.batch_size]
            with tf.GradientTape(persistent=True) as tape:
                generated_sample = g_model(LR_batch)
        
                logits_real = d_model(HR_batch)
                logits_fake = d_model(generated_sample)
        
                g_loss = generator.loss_function(fake_imgs=generated_sample, real_imgs=HR_batch, logits_fake=logits_fake,
                                               logits_real=logits_real)
                print("Generator Loss :", g_loss)
                d_loss = discriminator.loss_function(logits_fake=logits_fake, logits_real=logits_real)
                print("Discriminator Loss :", d_loss)
            g_grad = tape.gradient(g_loss, g_model.trainable_variables)
            self.optimizer.apply_gradients(zip(g_grad, g_model.trainable_variables))
        
            d_grad = tape.gradient(d_loss, d_model.trainable_variables)
            self.optimizer.apply_gradients(zip(d_grad, d_model.trainable_variables))
        
        return g_loss, d_loss


# def test():

# def eval_model():


if __name__ == "__main__":
    srgan = SRGAN()
    normal_lr, normal_hr = srgan.preprocess()
    g_loss, d_loss = srgan.train(normal_lr[:32], normal_hr[:32])
