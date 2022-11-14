# Point Map Neural Network

## Inputs and outputs

Input vector: `[x_distorted, y_distorted, k1, k2, k3, p1, p2]` where `(x_distorted, y_distorted)` is the 2D position of the camera in the distorted image and `(k1, k2, k3, p1, p2)` are the distortion parameters.

Output vector: `[x, y]` position of the undistorted image.

## Structure

3-layer fully connected neural network 2 hidden layers (32, 16) with ReLu activations and 1 output layer (2) with linear activation.

## Loss function

$$L = \frac{1}{n} \sum_i^n\left\| p_{\text{pred}} - p_{\text{true}} \right\|_2^2 + \gamma \sum_i \left\| W_i \right\|_2^2 $$

where $p_{\text{pred}}$ is the predicted undistorted position, $p_{\text{true}}$ is the ground truth point map, $W_i$ is the weight matrix of the $i$th layer, and $\gamma$ is the regularization parameter.


