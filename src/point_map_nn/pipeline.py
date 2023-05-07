# Pipeline class that encapsulates the entire pipeline for the point map nn

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import tensorflow as tf
import numpy as np
from itertools import product
from scipy.interpolate import interpn

from utils.images import read_image, write_image
from utils._paths import get_data_path

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing import paramType
    from numpy.typing import NDArray


class PointMapModelPipeline:
    """
    Point map neural network pipeline class

    Given initial setup (trained neural network model, interpolation model), the pipeline can be called as a function to distort images
    """

    point_map_model: tf.keras.models.Sequential
    interpolation_model_type: str

    def __init__(
        self, point_map_model: tf.keras.models.Sequential, interpolation_model_type: str
    ) -> None:
        self.point_map_model = point_map_model
        if interpolation_model_type not in [
            "linear",
            "cubic",
            "quintic",
            "nearest",
            "slinear",
            "pchip",
            "splinef2d",
        ]:
            raise ValueError(
                f"Interpolation model type {interpolation_model_type} not supported"
            )
        self.interpolation_model_type = interpolation_model_type

    def __call__(
        self,
        images: Union[list["NDArray"], "NDArray"],
        K: "paramType",
        P: "paramType",
        output_resolution: Optional[tuple[int, int]] = None,
    ):
        """
        Takes in either a single picture or a batch of pictures with output resolutions and returns the corresponding distorted pictures

        Resolution in the form (height, width)
        """
        if isinstance(images, list):
            raise NotImplementedError

        if output_resolution is None:
            output_resolution = images.shape[:2]

        # Grab output image normalized positions to feed into neural network
        scale = max(output_resolution)
        scale_input = max(images.shape[:2])

        x_min, x_max = -output_resolution[0] / scale, output_resolution[0] / scale
        y_min, y_max = -output_resolution[1] / scale, output_resolution[1] / scale
        x_min_input, x_max_input = (
            -images.shape[0] / scale_input,
            images.shape[0] / scale_input,
        )
        y_min_input, y_max_input = (
            -images.shape[1] / scale_input,
            images.shape[1] / scale_input,
        )
        x_range = np.linspace(x_min, x_max, output_resolution[0])
        y_range = np.linspace(y_min, y_max, output_resolution[1])
        x_range_input = np.linspace(x_min_input, x_max_input, images.shape[0])
        y_range_input = np.linspace(y_min_input, y_max_input, images.shape[1])

        # Run output point maps through neural network to get query points
        neural_network_input = np.empty((np.prod(output_resolution), 8))
        for i, (x, y) in enumerate(product(x_range, y_range)):
            neural_network_input[i, :] = x, y, x**2 + y**2, *K, *P

        query_points = self.point_map_model.predict(
            neural_network_input, batch_size=np.prod(output_resolution)
        )

        # Interpolate on all query points
        interpolated_points_B = interpn(
            (x_range_input, y_range_input),
            images[:, :, 0],
            query_points,
            method=self.interpolation_model_type,
            bounds_error=False,
            fill_value=0,
        )
        interpolated_points_G = interpn(
            (x_range_input, y_range_input),
            images[:, :, 1],
            query_points,
            method=self.interpolation_model_type,
            bounds_error=False,
            fill_value=0,
        )
        interpolated_points_R = interpn(
            (x_range_input, y_range_input),
            images[:, :, 2],
            query_points,
            method=self.interpolation_model_type,
            bounds_error=False,
            fill_value=0,
        )

        # Convert output point map to images
        return np.stack([interpolated_points_B, interpolated_points_G, interpolated_points_R], axis=-1).reshape(output_resolution[0], output_resolution[1], 3).astype(np.uint8)  # type: ignore


if __name__ == "__main__":
    from model import create_model

    img = read_image(get_data_path("images", "checkerboard.jpg"))
    model = create_model(loss="mse", reg=0)
    model.load_weights("model_weights/mse_point_map_no_reg.h5")
    model.summary()
    pmmp = PointMapModelPipeline(model, "nearest")
    distorted_img = pmmp(
        img,
        (0.01, 0.03, 0.02),
        (0.00, 0.00),
        output_resolution=(img.shape[0] * 3, img.shape[1] * 3),
    )
    write_image(distorted_img, "images/test_distorted_img.png")
