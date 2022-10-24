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