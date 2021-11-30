import tensorflow as tf


class Config:
    data_dir = '/Users/adidot/Downloads/archive/pokemon_jpg/'
    epochs = 5
    scale = 4
    batch_size = 16
    learning_rate = 1e-3
    beta = 0.9
    rescaling_factor = 0.006
    num_channels = 3
    image_height = 256
    image_width = 256
    input_shape = (batch_size, image_height, image_width, num_channels)
    input_shape_lr = (batch_size,image_height // scale, image_width // scale, num_channels)
    input_shape_hr = (image_height, image_width, num_channels)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
