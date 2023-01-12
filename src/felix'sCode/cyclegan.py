import cyclegan_utils
import pix2pix_utils
import tensorflow as tf

OUTPUT_CHANNELS = 3

height=256
width=256
unfixed_image_name= "test/img_distorted.png"
fixed_image_name= "test/img_fixed.png"
## ------------------------------------- Reading the image file -------------------------------------##
unfixed_image=pix2pix_utils.read_image(unfixed_image_name)
fixed_image=pix2pix_utils.read_image(fixed_image_name)
## ------------------------------------- Resizing the Image Files -------------------------------------##
unfixed_image=tf.image.resize(unfixed_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
fixed_image=tf.image.resize(fixed_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

unfixed_train,fixed_train=pix2pix_utils.load_image_train(unfixed_image,fixed_image)
unfixed_test,fixed_test=pix2pix_utils.load_image_test(unfixed_image,fixed_image)
unfixed_train=tf.expand_dims(unfixed_train, 0)
fixed_train=tf.expand_dims(fixed_train, 0)
unfixed_test=tf.expand_dims(unfixed_test, 0)
fixed_test=tf.expand_dims(fixed_test, 0)
## -------------------------------------- Tensorflow Data Pipeline  -------------------------------------------------##
## adding 1 dimension to the images for pix2pix training
train_dataset=tf.data.Dataset.from_tensor_slices((tf.expand_dims(unfixed_train, 0),tf.expand_dims(fixed_train, 0)))
test_dataset=tf.data.Dataset.from_tensor_slices((tf.expand_dims(unfixed_test, 0),tf.expand_dims(fixed_test, 0)))



## Intializing the generator and discriminator
generator_g = cyclegan_utils.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = cyclegan_utils.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = cyclegan_utils.discriminator(norm_type='instancenorm', target=False)
discriminator_y = cyclegan_utils.discriminator(norm_type='instancenorm', target=False)


