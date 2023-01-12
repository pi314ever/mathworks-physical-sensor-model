import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import cv2 as cv

## Global Variables
IMG_WIDTH = 256
IMG_HEIGHT = 256

## --------------------- Helper Function for loading an image ------------------##
def read_image(image_name):
	image = tf.io.read_file(image_name)
	image = tf.io.decode_jpeg(image)
	plt.imshow(image)
	return image


## --------------------- Helper Function for random jittering------------------- ##
def resize(unfixed, fixed, height, width):
    unfixed = tf.image.resize(unfixed, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    fixed = tf.image.resize(fixed, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return unfixed, fixed
def random_crop(unfixed, fixed):
    stacked_image = tf.stack([unfixed, fixed], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]
def normalize(unfixed, fixed): ## [-1,1]
    unfixed = (unfixed / 127) - 1
    fixed = (fixed / 127) - 1

    return unfixed, fixed
@tf.function()
def random_jitter(unfixed, fixed):
  # Resizing to 286x286
    unfixed, fixed = resize(unfixed, fixed, 286, 286)

    # Random cropping back to 256x256
    unfixed, fixed = random_crop(unfixed, fixed)

    if tf.random.uniform(()) > 0.5:
    # Random mirroring
        unfixed = tf.image.flip_left_right(unfixed)
        fixed = tf.image.flip_left_right(fixed)

    return unfixed, fixed
def load_image_train(unfixed,fixed):
    unfixed, fixed = random_jitter(unfixed, fixed)
    unfixed, fixed = normalize(unfixed, fixed)

    return unfixed, fixed
def load_image_test(unfixed,fixed):
    unfixed,fixed=resize(unfixed, fixed, 256, 256)
    unfixed, fixed = normalize(unfixed, fixed)

    return unfixed, fixed

## --------------------- Constructing the Generator ------------------- ##
def downsample(filters, size ,apply_batchnorm= True):
    initializer = tf.random_normal_initializer(0.,0.02)
    model= tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    return model
def upsample(filters,size,apply_dropout=False):
    initializer = tf.random_normal_initializer(0.,0.02)
    model= tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2DTranspose(filters,size,strides=2,padding='same',
                                kernel_initializer=initializer,use_bias=False)
    )
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.ReLU())
    return model
OUTPUT_CHANNELS = 3
def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    model = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        model = down(model)
        skips.append(model)
    
    skips = reversed(skips[:-1]) ## ignoring the first layer and starting from 
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        model = up(model)
        model = tf.keras.layers.Concatenate()([model, skip])

    model = last(model)

    return tf.keras.Model(inputs=inputs, outputs=model)


## --------------------- Generator Loss ------------------------------------- ##
# Define the generator loss
LAMBDA = 100
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
    ## gan loss= sum(log(1-Disc_generated_output))
    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    ## l1_loss= sum()
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

## --------------------- Constructing the Discriminator ------------------- ##
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    unfixed = tf.keras.layers.Input(shape=[256, 256, 3], name='unfixed_image')
    fixed = tf.keras.layers.Input(shape=[256, 256, 3], name='fixed_image')

    x = tf.keras.layers.concatenate([unfixed, fixed])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[unfixed, fixed], outputs=last)
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)
    ## bce(1,D(Target,Input))
    generated_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)
    ## bce(0,D(Generted_image, Input_image))
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
## ---------------------------- Optimizer ------------------------------------------------------ ##
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
def generate_images(model, test_input, target):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    return test_input[0],target[0],prediction[0]
## ---------------------------- Training Step ------------------------------------------------------ ##
@tf.function
def train_step(input_image, target, generator,discriminator, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        ## generate an image given the original unfixed image
        gen_output= generator(input_image,training=True)
        ## D(input_image, target)
        disc_real_output=discriminator([input_image,target],training=True)
        ## D(input_image,gen_output) ## fake target image
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        ## Obtain the losses from generator_loss(disc_generated output,gen_output,target)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        ## discrimantor loss given disc realoutput and disc generated output
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    # gradient of generator with respect to all the variables of generator
    generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
    # gradient of dicriminator with respect to all the variables of generator

    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
    ## apply gradients with(grads,[var1,var2])
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
## ---------------------------- Training Step ------------------------------------------------------ ##
def fit(train_ds, test_ds, generator,discriminator,steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if step%1000==0:
            print(step)
            generate_images(generator, example_input, example_target)


        train_step(input_image, target, generator, discriminator,step)

     
