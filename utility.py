import tensorflow as tf
import os
import time

from matplotlib import pyplot as plt
from IPython import display

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2

  # Assign the first half of Dimension 1 of the image to real_image
  real_image = image[:, :w, :]
  # Assign the second half of Dimension 1 of the image to input_image
  input_image = image[:, w:, :]
  # Ensure that the pixel values are represented as TF 32-bit floating point numbers
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image

def random_crop(input_image, real_image, img_height, img_width):
  # Concatenate/stack the input and real images (in that order) along the 0 dimension
  stacked_image = tf.stack([input_image, real_image], axis=0) 
  # Crop an img_height by img_width area out of the stacked_image
  # Return cropped_image as an array with 2 elements in its 0 dimension, 
  # one for the input and one for the real image
  cropped_image = tf.image.random_crop(stacked_image, 
                                       size=[2, img_height, img_width, 3])
  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
  """
  Normalizes input images to [-1, 1]
  """
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1 

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  ### START CODE HERE ###
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286) 

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image, 256, 256)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image) 
  ### END CODE HERE ###
  return input_image, real_image

def load_image_train(image_file):
  ### START CODE HERE ###
  # Load
  input_image, real_image = load(image_file)
  # Random jitter
  input_image, real_image = random_jitter(input_image, real_image)
  # Normalize
  input_image, real_image = normalize(input_image, real_image)
  ### END CODE HERE ###
  return input_image, real_image

def load_image_test(image_file):
  ### START CODE HERE ###
  # Load
  input_image, real_image = load(image_file)
  # Resize to IMG_HEIGHT by IMG_WIDTH
  input_image, real_image = resize(input_image, real_image, 256, 256)
  # Normalize
  input_image, real_image = normalize(input_image, real_image)
  ### END CODE HERE
  return input_image, real_image  

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()   