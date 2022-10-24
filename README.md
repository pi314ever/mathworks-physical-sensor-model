# Applying Machine Learning for the Development of Physical Sensor Models in Game Engine Environment

Contributors: Daniel Huang, Felix Meng, and Jason Zhu

## Introduction

This project aims to generate a model which can calculate distortion effects efficiently and accurately given distortion parameters and an undistorted image.

## Dataset and Data Pipeline

TODO: Complete list of dataset pipeline items
- [ ] Real images dataset
  - [ ] Gather a dataset of undistorted images
  - [ ] Create distortion pipeline to generate distorted labels
  - [ ] Package data into format to be used by models
- [ ] Synthetic image dataset
  - [ ] Generate base library of shapes and textures
  - [ ] Generate combination library
  - [ ] Create massive dataset of undistorted images
  - [ ] Pre-calculate distortion $(x_{\text{distorted}}, y_{\text{distorted}})$ for each pixel $(x, y)$ in the undistorted image

## Models

TODO: Complete list of models to be used.
- [ ] Deep NN with some regression loss
- [ ] Selective supersampling and interpolation
- [ ] Cycle-GANs
- [ ] Traditional distortion algorithms (e.g. OpenCV) as baseline


## Useful Links

- [MATLAB onramp](https://www.mathworks.com/learn/tutorials/matlab-onramp.html)
- [Simulink onramp](https://www.mathworks.com/learn/tutorials/simulink-onramp.html)
- [Machine Learning onramp](https://www.mathworks.com/learn/tutorials/machine-learning-onramp.html)
- [Deep Learning onramp](https://www.mathworks.com/learn/tutorials/deep-learning-onramp.html)

Other links

- https://github.com/mathworks/MathWorks-Excellence-in-Innovation/tree/main/projects/Applying%20Machine%20Learning%20for%20the%20Development%20of%20Physical%20Sensor%20Models%20in%20Game%20Engine%20Environment
- https://www.mathworks.com/help/vision/ug/camera-calibration.html
- https://www.mathworks.com/help/vision/ref/undistortimage.html
- https://www.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
- https://www.mathworks.com/help/driving/ref/simulation3dcamera.html
