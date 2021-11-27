from keras import backend
from keras.applications.vgg19 import VGG19
from keras.losses import mean_squared_error as mse

# Discriminator Loss
class Loss:

	def discriminator_loss(logits_fake: tf.Tensor, logits_real: tf.Tensor) -> tf.Tensor:
		D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
		D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
		return D_loss

	  # Generator Loss
	def generator_loss(fake_imgs: tf.Tensor, real_imgs: tf.Tensor,logits_fake: tf.Tensor, logits_real: tf.Tensor,i=5,j=4) -> tf.Tensor:
		model = VGG19(weights='imagenet')
		model = backend.function([model.layers[0].input], [model.layers[i*j].output])
		vgg_input_features = model(real_imgs)[0]
		vgg_target_features = model(fake_imgs)[0]

		content_loss = mse(vgg_input_features,vgg_target_features)
		adversarial_loss = 1e-3 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
		feature_dim = vgg_input_features.shape[1]*vgg_input_features.shape[2]
		perceptual_loss = (1/feature_dim)*content_loss + adversarial_loss

		return perceptual_loss