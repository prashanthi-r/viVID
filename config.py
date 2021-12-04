import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

class Config:
    #data_dir = '/Users/rugvedmavidipalli/Downloads/archive/pokemon_jpg/'
    # for data dir now you have to give the full path where images are so for pokemon: pokemon_jpg/pokemon_jpg
    data_dir = '/Users/rugvedmavidipalli/Downloads/DIV2K_train_HR/'
    epochs = 1
    scale = 4
    batch_size = 16
    learning_rate = 1e-3
    beta = 0.9
    rescaling_factor = 0.006
    num_channels = 3
    image_height = 256
    image_width = 256
    input_shape = (image_height, image_width, num_channels)
    input_shape_lr = (image_height // scale, image_width // scale, num_channels)
    input_shape_hr = (image_height, image_width, num_channels)
    optimizer = tf.keras.optimizers.Adam(PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5]))
