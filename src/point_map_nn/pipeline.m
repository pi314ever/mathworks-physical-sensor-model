% Recreating pipeline in matlab
clc;clear;close all
conda_dir = "C:\Users\pilot\miniconda3\envs\xplore\python.exe";

pyenv("Version",conda_dir);

model_layers = importKerasLayers('src\point_map_nn\model_weights\mse_point_map_no_reg.h5', 'ImportWeights',true);
model_layers = layerGraph(model_layers);

placeholders = findPlaceholderLayers(model_layers);

dense_1 = fullyConnectedLayer(16, ...
    "Weights", model_layers.Layers(2).Weights.kernel, ...
    "Bias", model_layers.Layers(2).Weights.bias);
dense_2 = fullyConnectedLayer(16, ...
    "Weights", model_layers.Layers(3).Weights.kernel, ...
    "Bias", model_layers.Layers(3).Weights.bias);
dense_3 = fullyConnectedLayer(16, ...
    "Weights", model_layers.Layers(4).Weights.kernel, ...
    "Bias", model_layers.Layers(4).Weights.bias);
dense_4 = fullyConnectedLayer(16, ...
    "Weights", model_layers.Layers(5).Weights.kernel, ...
    "Bias", model_layers.Layers(5).Weights.bias);
relu = reluLayer();

model_layers = replaceLayer(model_layers, 'dense', [dense_1, relu]);
model_layers = replaceLayer(model_layers, 'dense_1', [dense_2, relu]);
model_layers = replaceLayer(model_layers, 'dense_2', [dense_3, relu]);
model_layers = replaceLayer(model_layers, 'dense_3', [dense_4, relu]);
model_layers = model_layers.removeLayers('RegressionLayer_dense_4');

net = dlnetwork(model_layers);

test_image = imread("data\images\render-checkerboard1.jpg");
%%
distorted_img = distort(test_image);

imshow(distorted_img);

function distorted_img = distort(img, K, P)
    % Assume output shape same as input shape
    shape = size(img);
    max_dim = max(shape);
    x_min 

end






