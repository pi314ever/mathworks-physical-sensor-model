# Applying Machine Learning for the Development of Physical Sensor Models in Game Engine Environment

Contributors: Daniel Huang, Felix Meng, and Jason Zhu

## Abstract

This project aims to generate a model which can calculate distortion effects efficiently and accurately given distortion parameters and an undistorted image. There are use cases such as full-simulation training for self-driving cars, where distortion model of the camera needs to be accurately and efficiently calculated from a simulated environment. Current un-distortion algorithms utilize standard iterative methods and are not optimized for real-time applications. We present two approaches to this problem: a GAN approach (led by Felix Meng) and a function approximation method (led by Daniel Huang). The GAN method showed promising results, but needed to be trained specifically per distortion parameter and cannot be generalized. The function approximation method was less visually accurate due to the point-wise nature of the model. However, it is able to generalize to different distortion parameters and is much more efficient than the GAN method.

## Setup

**Prerequisites**
- `poetry`: Python package management tool
- `python3.9`: Python 3.9
- `cuda` (optional): GPU support

**Steps**
1. Install `poetry` packages with `poetry install`
2. Test environment setup with `poetry run pytest`

## Dataset and Data Pipeline

### Point map dataset

Dataset is not included by default (git ignored). To generate the dataset, run `python src/data/generate_point_maps.py`. This will generate the dataset in `data/point_maps/` directory.

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
