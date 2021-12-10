import tensorflow as tf

class Config:
    data_dir = '/content/drive/MyDrive/viVID/DIV2K_train_HR'
    pretrain_generator = False
    pretrain_generator_epochs = 50
    starting_epoch = 1740
    generator_weights_path = '/content/drive/MyDrive/viVID/model_ckpts/srgan/srgan/gen1740.h5'
    epochs = 4000
    scale = 4
    batch_size = 64
    beta = 0.9
    rescaling_factor = 0.006
    num_channels = 3
    image_height = 96
    image_width = 96
    input_shape = (image_height, image_width, num_channels)
    input_shape_lr = (image_height // scale, image_width // scale, num_channels)
    input_shape_hr = (image_height, image_width, num_channels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,
                                         beta_1=0.5)
    GCP = False
