import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Dropout, ReLU, Input

import os
import time

from matplotlib import pyplot as plt
from IPython import display

from utility import generate_images
from models import Generator, Discriminator, generator_loss, discriminator_loss

generator = Generator(3)
discriminator = Discriminator()

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

EPOCHS = 1

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Define the optimizers and checkpoint_saver
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(input_image, target, epoch):
  ### START CODE HERE ###
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

    # Apply the generator to the input image, with training=True
    gen_output = generator(input_image, training=True)

    # Apply the discriminator to the (concatenated) input image and target, with training=True
    disc_real_output = discriminator([input_image, target], training=True)
    # Apply the discriminator to the (concatenated) input image and generated output, with training=True
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    # Compute the generator loss(es)
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    # Compute the discriminator loss(es)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  # Compute the gradient of the total generator loss with respect to the generator's trainable_variables
  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  # Compute the gradient of the total discriminator loss with respect to the discriminator's trainable_variables
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  # Apply the generator gradients to the generator_optimizer
  # Zip the gradients and variables to do this
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  
  # Apply the discriminator gradients to the discriminator_optimizer
  # Zip the gradients and variables to do this
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  ### END CODE HERE ###
  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)



  ### END CODE HERE ###
  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)