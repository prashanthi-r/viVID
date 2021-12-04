from matplotlib import image
import numpy as np
from PIL import Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale, resize, downscale_local_mean
from math import log10, sqrt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def eval(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 0, 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return mse, psnr


def load_image(path):
    return io.imread(path)


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))


def get_data(data_dir, scale):
    print(f"{'---' * 10} PREPROCESSING STARTED {'---' * 10}")
    lr_images = []
    hr_images = []
    normal_lr_images = []
    normal_hr_images = []

    data_dir = pathlib.Path(data_dir)
    img_list = list(data_dir.glob('*.jpg'))
    if len(img_list) <= 0:
        img_list = list(data_dir.glob('*.png'))
    image_count = len(img_list)
    print(f"| NUMBER OF IMAGES: {image_count} |")
    print(f"| SHAPE OF IMAGES: {load_image(img_list[0]).shape} |")

    high_res_h, high_res_w, channel = (256, 256, 3)
    # if load_image(img_list[0]).shape != load_image(img_list[0]).shape:

    low_res_h = high_res_h // scale
    low_res_w = high_res_w // scale
    print(
        f"| high_res_h: {high_res_h} | \n| high_res_w: {high_res_w} | \n| low_res_h: {low_res_h} | \n| low_res_w :{low_res_w} |\n")
    count = 0
    for i in img_list[:]:
        img = resize(load_image(i), (high_res_h, high_res_w), preserve_range=True)
        img_lr = resize(img, (low_res_h, low_res_w), preserve_range=True)
        normalization_layer_hr = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)
        normal_img_hr = normalization_layer_hr(img)
        normalization_layer_lr = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        normal_img_lr = normalization_layer_lr(img_lr)
        lr_images.append(img_lr)
        hr_images.append(img)
        normal_lr_images.append(normal_img_lr)
        normal_hr_images.append(normal_img_hr)
        count += 1
        print('\r', 'Preprocessing: {0:.2f} %'.format((count / image_count) * 100), end='')
    print(f"\n{'---' * 10} PREPROCESSING COMPLETED {'---' * 10}")
    return np.array(normal_lr_images), np.array(normal_hr_images)


def preprocessing_demo(data_dir, scale, demo_size):
    print(f"{'---' * 10} DEMO {'---' * 10}")
    data_dir = pathlib.Path(data_dir)
    img_list = list(data_dir.glob('*/*.jpg'))

    high_res_h, high_res_w, _ = load_image(img_list[0]).shape
    low_res_h = high_res_h // scale
    low_res_w = high_res_w // scale
    print(high_res_h, high_res_w, low_res_h, low_res_w)

    for i in range(demo_size):
        img = load_image(img_list[i])
        # print(np.min(img),np.max(img))
        # img_lr = resize(img, (low_res_h,low_res_w))
        img_lr = resize(img, (low_res_h, low_res_w))
        img_hr_naive = resize(img_lr, (high_res_h, high_res_w))

        fig, axes = plt.subplots(nrows=1, ncols=3)
        ax = axes.ravel()
        ax[0].imshow(img)
        ax[0].set_title("HR image")
        ax[1].imshow(img_lr)
        ax[1].set_title("LR")
        ax[2].imshow(img_hr_naive)
        ax[2].set_title("Naive HR")
        plt.tight_layout()
        plt.show()
