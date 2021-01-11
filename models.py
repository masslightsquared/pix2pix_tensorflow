import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Dropout, ReLU, Input

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def downsample(filters, size, apply_batchnorm=True):

  # Initialize the parameters so they are normally distributed, 
  # with a mean of 0 and standard deviation of 0.02
  initializer = tf.random_normal_initializer(0., 0.02)
  
  # Instantiate a sequential block
  result = Sequential()
  
  # Add a 2D convolutional layer
  # Use a stride of 2, same padding, and the initializer defined above
  # DON'T use a bias vector
  result.add(Conv2D(filters, size, strides=2, padding='same',
                    kernel_initializer=initializer, use_bias=False))
  
  # Add batch normalization, if appropriate
  if apply_batchnorm:
    result.add(BatchNormalization())
  
  # Add a leaky ReLU activation
  result.add(LeakyReLU())
    
  return result


def upsample(filters, size, apply_dropout=False):

  # Initialize the parameters so they are normally distributed, 
  # with a mean of 0 and standard deviation of 0.02
  initializer = tf.random_normal_initializer(0., 0.02)

  # Instantiate a sequential block
  result = Sequential()
  
  # Add a 2D transposed convolutional layer
  # Use a stride of 2, same padding, and the initializer defined above
  # DON'T use a bias vector
  result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  
  # Add batch normalization
  result.add(BatchNormalization())

  # Add dropout with a probability of 0.5, if appropriate
  if apply_dropout:
      result.add(Dropout(0.5))
  
  # Add a ReLU activation
  result.add(ReLU())

  return result  

def Generator(OUTPUT_CHANNELS):

  # Define a keras input layer with shape [256,256,3]
  inputs = Input(shape=[256,256,3])

  # Define a stack of downsampling blocks
  # Don't apply batch normalization to the first block, but do apply it to the rest
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (batch size, 128, 128, 64)
    downsample(128, 4), # (batch size, 64, 64, 128)
    downsample(256, 4), # (batch size, 32, 32, 256)
    downsample(512, 4), # (batch size, 16, 16, 512)
    downsample(512, 4), # (batch size, 8, 8, 512)
    downsample(512, 4), # (batch size, 4, 4, 512)
    downsample(512, 4), # (batch size, 2, 2, 512)
    downsample(512, 4), # (batch size, 1, 1, 512)
  ]

  # Define a stack of upsampling blocks
  # Apply dropout regularization to the first 3 blocks, but don't apply it to the rest
  up_stack = [
    upsample(512, 4, apply_dropout=True), # (batch size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (batch size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (batch size, 8, 8, 1024)
    upsample(512, 4), # (batch size, 16, 16, 1024)
    upsample(256, 4), # (batch size, 32, 32, 512)
    upsample(128, 4), # (batch size, 64, 64, 256)
    upsample(64, 4), # (batch size, 128, 128, 128)
  ]

  # Initialize the parameters so they are normally distributed, 
  # with a mean of 0 and standard deviation of 0.02
  initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
  
  # Define the final layer as a 2D transposed convolution with 
  # OUTPUT_CHANNELS filters, kernel size of 4, stride of 2, same padding, 
  # the initializer defined above, and tanh activation
  last = Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                         kernel_initializer=initializer, activation='tanh') # (batch size, 256, 256, 3)

  # Now that we've defined the generator's layers, let's assemble them into a network 
  # with the keras functional API

  # Start with the input layer
  x = inputs

  # Downsampling through the model
  skips = []
  # Loop through the downsampling blocks
  for down in down_stack:
    # Apply the current downsampling block
    x = down(x) 
    # Append the downsampled data to the skips list
    # We'll use it for upsampling later
    skips.append(x) 
  
  # Take all but the last element in the skips list and reverse them
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  # Loop through the zipped upsampling stack and skips list
  for up, skip in zip(up_stack, skips):
    # Apply the current upsampling block
    x = up(x)
    # Concatenate the upsampled data and the appropriately downsampled data from the skips list
    x = tf.keras.layers.Concatenate()([x, skip])

  # Apply the last layer
  x = last(x)
  ### END CODE HERE ###
  return tf.keras.Model(inputs=inputs, outputs=x)  

def generator_loss(disc_generated_output, gen_output, target):

  # Use loss_object to compute the loss between the discriminator-generated output
  # and a tensor of ones of the same shape
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Compute the mean absolute error between the target and the generator's output
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  # Compute the total generator loss
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  
  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  ### START CODE HERE ###

  # Initialize the parameters so they are normally distributed, 
  # with a mean of 0 and standard deviation of 0.02
  initializer = tf.random_normal_initializer(0., 0.02) 

  # Define the input and target layers as keras input layers with shape [256,256,3]
  # Use 'input_image' for the name
  inp = Input(shape=[256, 256, 3], name='input_image')
  # Use 'target_image' for the name
  tar = Input(shape=[256, 256, 3], name='target_image')

  # Assemble the discriminator network sequentially, but using the keras functional API
  
  # Concatenate the input and target
  x = tf.keras.layers.concatenate([inp, tar]) # (batch size, 256, 256, channels*2)

  # Apply 3 downsampling blocks
  # Use the output sizes to determine the first 2 arguments of each block
  # Don't apply batch normalization to the first block, but do apply it to the others
  down1 = downsample(64, 4, False)(x) # (batch size, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (batch size, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (batch size, 32, 32, 256)

  # Apply 2D zero padding with a padding width of 1 (default)
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (batch size, 34, 34, 256)

  # Apply a 2D convolution with 512 output channels, kernel size of 4, stride of 1, 
  # the initializer defined above, and no bias parameters
  conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                use_bias=False)(zero_pad1) # (batch size, 31, 31, 512)

  # Apply batch normalization
  batchnorm1 = BatchNormalization()(conv)

  # Apply leaky ReLU activation
  leaky_relu = LeakyReLU()(batchnorm1)

  # Apply 2D zero padding with a padding width of 1 (default)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (batch size, 33, 33, 512)

  # Apply a 2D convolution with 1 output channel, kernel size of 4, stride of 1, and 
  # the initializer defined above
  last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (batch size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)  

def discriminator_loss(disc_real_output, disc_generated_output):

  # Use loss_object to compute the loss between the real image(s)
  # and a tensor of ones of the same shape
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  # Use loss_object to compute the loss between the discriminator-generated output
  # and a tensor of ones of the same shape
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  # Sum the two losses computed above
  total_disc_loss = real_loss + generated_loss

  return total_disc_loss  

