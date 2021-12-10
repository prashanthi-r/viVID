from matplotlib import image
import numpy as np
from PIL import Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.feature_extraction import image
from math import log10, sqrt
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def eval(original, compressed):
    """
    This calculates the mse and psnr for two images 

    Args:
        original image (numpy array: h,w,c)
        test image (numpy array: h,w,c)

    Returns:
        mse: mean square error
        psnr: peak signal to noise ratio
    """ 
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 0, 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return mse, psnr

# def show_images(images, fig_name , res, epoch, loc):
#     if res == 'high':
#         #-1 to 1
#         images = ((np.array(images) + 1) * 127.5).astype(np.uint8)
#     elif res == 'low':
#         # 0 to 1
#         images = (np.array(images)*255).astype(np.uint8)

#     images =  np.array(images)
#     print(f"fig {fig_name}\nimg min:{min(np.reshape(images[0],[-1]))} max:{max(np.reshape(images[0],[-1]))}")
#     fig = plt.figure(figsize=(1, images.shape[0]))
#     print(f"fig {fig_name}\nimg min:{min(np.reshape(images[0],[-1]))} max:{max(np.reshape(images[0],[-1]))}")
#     gs = gridspec.GridSpec(1, images.shape[0])
#     gs.update(wspace=0.05, hspace=0.05)
#     fig.set_size_inches(7.5, 7.5)
#     for i, img in enumerate(images):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(img)
#         plt.title(fig_name)
#     #plt.savefig(loc+'/'+fig_name+str(epoch)+'.png')
#     return

def load_image(path):
    '''
    This function reads an input image
    Args:
        path(str): path to the input image
    Returns:
        image(numpy array)
    '''
    return io.imread(path)

def save_image(path, im):
    '''
    This function saves an input image
    Args:
        path(str): path to the input image
        im (numpy array): image to be saved
    '''
    return io.imsave(path, img_as_ubyte(im.copy()))

def get_data(data_dir, scale):
    '''
    This function is obsolete, replaved with get_data_clip
    Args:
        data_dir(str): path to the input images
        scale(int): factor to scale the input image (2x,3x,4x)
    Returns:
        normal_lr_images(numpy array): normalised low resolution images
        normal_hr_images(numpy array): normalised high resolution images
    '''
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

    high_res_h, high_res_w, channel = (96, 96, 3)
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

def get_data_clip(data_dir, scale, patch_size =96, seed=5, patches_count=1, gray_scale = False):
    '''
    This function loads the images from the data directory and selects patches_count number of random patches of 
    size (patch_size x patch_size) from the images and returns normalsied low res and high res images
    Args:
        data_dir(str): path to the input images
        scale(int): factor to scale the input image (2x,3x,4x)
        patch_size(int): size of the patches to pick from the images 
        seed(int): random seed to reproduce the output
        patch_count(int): number of patches to extract from each image
        gray_scale(boolean): future work, option to convert images to grayscale an train images
    Returns:
        normal_lr_images(numpy array): normalised low resolution images
        normal_hr_images(numpy array): normalised high resolution images
    '''
    print(f"{'---' * 10} PREPROCESSING WIHT CLIP STARTED {'---' * 10}")
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

    high_res_h, high_res_w, channel = (96, 96, 3)
    # if load_image(img_list[0]).shape != load_image(img_list[0]).shape:

    low_res_h = high_res_h // scale
    low_res_w = high_res_w // scale
    print(
        f"| high_res_h: {high_res_h} | \n| high_res_w: {high_res_w} | \n| low_res_h: {low_res_h} | \n| low_res_w :{low_res_w} |\n")
    count = 0
    for i in img_list[:]:
        
        patches = image.extract_patches_2d(load_image(i), (patch_size, patch_size), max_patches=patches_count,random_state=seed)
        for j in patches:
            img_lr = resize(j, (low_res_h, low_res_w), preserve_range=True)
            normalization_layer_hr = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)
            normal_img_hr = normalization_layer_hr(j)
            normalization_layer_lr = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
            normal_img_lr = normalization_layer_lr(img_lr)
            lr_images.append(img_lr)
            hr_images.append(j)
            normal_lr_images.append(normal_img_lr)
            normal_hr_images.append(normal_img_hr)
        count += 1
        print('\r', 'Preprocessing: {0:.2f} %'.format((count / image_count) * 100), end='')
    print(f"\n{'---' * 10} PREPROCESSING COMPLETED {'---' * 10}")
    return np.array(normal_lr_images), np.array(normal_hr_images)

# how to call:
# def preprocess(self):
#         return get_data_clip(self.dataset_directory, self.scalling_factor, patch_size =96, seed=5, patches_count=3, gray_scale = False)


def reconstruct(blocks, block_size, inp_img_w = 1356, inp_img_h = 2040):
    '''
    This function reconstructs the image from patches. Works inverse of deconstruct.
    Args:
        blocks(numpy array): (no of blocks, h,w,c) blocks os iamges 
        block size(int): size of the blocks
        inp_img_w(int): width of output image
        inp_img_h(int): height of output image
    Returns:
        image_recons(numpy ndarray): constructed input images
    '''
    rows = np.array(np.split(blocks[:,:,:,],inp_img_w/block_size,axis = 0))
    cols = np.array([np.hstack((rows[x,:,:,:,:])) for x in range(rows.shape[0])])
    image_recons = np.array([np.vstack((cols[:,:,:,:])) for x in range(cols.shape[0])])[0]
    return image_recons

def deconstruct(image, patch_size):
    '''
    This function converts the image into smaller patches. Works inverse of reconstruct.
    Args:
        image(numpy array): image 
        patch_size(int): size of the blocks
    Returns:
        blocks(numpy ndarray): [number of blocks, block_width, block_height, channel]
        h: height of the cropped image
        w: weight of the cropped image
        c: channel of the cropped image
    '''
    h,w,c = image.shape
    h = (h//patch_size)*patch_size
    w = (w//patch_size)*patch_size
    image = image[:h,:w,:]
    blocks = np.array([image[i:i+patch_size, j:j+patch_size]  for i in range(0,h,patch_size) for j in range(0,w,patch_size) ])
    return blocks, h, w, c


def test_video(generator, data_dir, scale, patch_size ):
    '''
    This function runs SRGAN- on a video file, it takes the input feed video from teh web cam,
    reduces the resolution of the frame, breaks it into patches and passes it though SRGAN to get
    high resolution patches. It then merges the patches into one image, and merges the frames 
    back into output video.
    Args:
        generator(tensorflow model): generator model 
        scale(int): factor to downscale low resolution image
        patch_size(int): patch size
    '''
    print(f"{'---' * 10} TEST VIDEO  {'---' * 10}")
    normal_lr_images = []
    vid = cv2.VideoCapture(0)
    high_res_h, high_res_w, channel = (96, 96, 3)

    low_res_h = high_res_h // scale
    low_res_w = high_res_w // scale
  
    while(True):
        normal_lr_images = []
        ret, frame = vid.read()
        frame = cv2.resize(frame,(96*7,96*5))
        blocks,h,w,c = deconstruct(frame, patch_size)
        for j in blocks:
            img_lr = resize(j, (low_res_h, low_res_w), preserve_range=True)
            normalization_layer_lr = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
            normal_img_lr = normalization_layer_lr(img_lr)
            normal_lr_images.append(normal_img_lr)
        normal_lr_images_gen = np.array(normal_lr_images)
        generated_samples = generator(normal_lr_images_gen, training=True)
        generated_samples = np.array([((np.array(x) + 1) * 127.5).astype(np.uint8) for x in generated_samples])        
        recon_img = reconstruct(generated_samples, patch_size, h,w)
        recon_img = cv2.blur(recon_img,(4,4))
        inp_img = reconstruct(np.array(normal_lr_images), patch_size/scale, h/scale,w)
        inp_img = resize(inp_img, (recon_img.shape), preserve_range=True)
        cv2.imshow('output hr', recon_img)
        cv2.imshow('input lr', inp_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

    

def test_preprocessing(generator, data_dir, scale, patch_size ):
    '''
    This function runs SRGAN- on an image file, it takes the input iamge, reduces the resolution of the image, 
    breaks it into patches and passes it though SRGAN to get high resolution patches. 
    It then merges the patches into one image.
    Args:
        generator(tensorflow model): generator model 
        data_dir(str): data directory of the test image
        scale(int): factor to downscale low resolution image
        patch_size(int): patch size
    '''
    print(f"{'---' * 10} PREPROCESSING STARTED {'---' * 10}")
    normal_lr_images = []
    op_images = []
    in_images=[]
    data_dir = pathlib.Path(data_dir)
    img_list = list(data_dir.glob('*.jpg'))
    if len(img_list) <= 0:
        img_list = list(data_dir.glob('*.png'))
    image_count = len(img_list)
    print(f"| NUMBER OF IMAGES: {image_count} |")
    print(f"| SHAPE OF IMAGES: {load_image(img_list[0]).shape} |")

    high_res_h, high_res_w, channel = (96, 96, 3)
    low_res_h = high_res_h // scale
    low_res_w = high_res_w // scale
    print(
        f"| high_res_h: {high_res_h} | \n| high_res_w: {high_res_w} | \n| low_res_h: {low_res_h} | \n| low_res_w :{low_res_w} |\n")
    count = 0
    for i in img_list[:]:
        blocks,h,w,c = deconstruct(load_image(i), patch_size)
        for j in blocks:
            img_lr = resize(j, (low_res_h, low_res_w), preserve_range=True)
            normalization_layer_lr = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
            normal_img_lr = normalization_layer_lr(img_lr)
            normal_lr_images.append(normal_img_lr)
        normal_lr_images = np.array(normal_lr_images)
        generated_samples = generator(normal_lr_images, training=True)
        generated_samples = np.array([((np.array(x) + 1) * 127.5).astype(np.uint8) for x in generated_samples])
        recon_img = reconstruct(generated_samples, patch_size, h,w)
        inp_img = reconstruct(normal_lr_images, patch_size/scale, h/scale,w)
        inp_img = resize(inp_img, (recon_img.shape), preserve_range=True)
        op_images.append(recon_img)
        in_images.append(inp_img)
    return np.array(op_images), np.array(in_images)