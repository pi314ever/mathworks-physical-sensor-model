import pix2pix_utils
import tensorflow as tf
import cv2 as cv 
import matplotlib.pyplot as plt

## Global Variables
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
## ------------------------------------- Testing Code-------------------------------------------------##
# plt.imshow(unfixed_image)
# plt.show()
# plt.imshow(fixed_image)
# plt.show()

## -------------------------------------- Loading Image -------------------------------------------------##
unfixed_train,fixed_train=pix2pix_utils.load_image_train(unfixed_image,fixed_image)
unfixed_test,fixed_test=pix2pix_utils.load_image_test(unfixed_image,fixed_image)
# print(unfixed_test.shape)
# plt.imshow(unfixed_test)
# plt.show()
# plt.imshow(fixed_test)
# plt.show()
## -------------------------------------- Tensorflow Data Pipeline  -------------------------------------------------##
## adding 1 dimension to the images for pix2pix training
unfixed_train=tf.expand_dims(unfixed_train, 0)
fixed_train=tf.expand_dims(fixed_train, 0)
unfixed_test=tf.expand_dims(unfixed_test, 0)
fixed_test=tf.expand_dims(fixed_test, 0)

train_dataset=tf.data.Dataset.from_tensor_slices((tf.expand_dims(unfixed_train, 0),tf.expand_dims(fixed_train, 0)))
test_dataset=tf.data.Dataset.from_tensor_slices((tf.expand_dims(unfixed_test, 0),tf.expand_dims(fixed_test, 0)))

generator = pix2pix_utils.Generator()
discriminator = pix2pix_utils.Discriminator()

pix2pix_utils.fit(train_dataset,test_dataset,generator,discriminator,40000)
