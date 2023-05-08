# Pipeline class that encapsulates the entire pipeline for the point map nn

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from itertools import product
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from scipy.interpolate import interpn
from utils._paths import get_data_path
from utils.images import read_image, write_image

if TYPE_CHECKING:
    from model import Model
    from numpy.typing import NDArray
    from utils.typing import paramType


class PointMapModelPipeline:
    """
    Point map neural network pipeline class

    Given initial setup (trained neural network model, interpolation model), the pipeline can be called as a function to distort images
    """

    point_map_model: "Model"
    interpolation_model_type: str

    def __init__(self, point_map_model: "Model", interpolation_model_type: str) -> None:
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
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        x_mesh = x_mesh.flatten()
        y_mesh = y_mesh.flatten()
        vstack = np.vstack((x_mesh, y_mesh))
        vstack = vstack.T

        K_tf = np.array(K)
        P_tf = np.array(P)

        query_points = self.point_map_model.predict(vstack, K_tf, P_tf)

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

        # Convert output point map to images, not sure why the axes are swapped
        return np.swapaxes(np.stack([interpolated_points_B, interpolated_points_G, interpolated_points_R], axis=-1).reshape(output_resolution[1], output_resolution[0], 3).astype(np.uint8), 0, 1)  # type: ignore


if __name__ == "__main__":
    from argparse import ArgumentParser

    from model import CombinedPMNN, SeparatePMNN

    parser = ArgumentParser()
    parser.add_argument("model_paths", type=str, help="Path to model, from ")
    parser.add_argument(
        "--image_path", type=str, default=get_data_path("images", "checkerboard.jpg")
    )
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    img = read_image(args.image_path)
    model_paths = args.model_paths.split(",")
    if len(model_paths) == 1:
        model = CombinedPMNN(model_paths[0])
    elif len(model_paths) == 2:
        model = SeparatePMNN(model_paths[0], model_paths[1])
    else:
        raise ValueError("Invalid number of model paths")

    pmmp = PointMapModelPipeline(model, "nearest")
    distorted_img = pmmp(
        img,
        (0.01, 0.03, 0.02),
        (0.00, 0.00),
        output_resolution=(img.shape[0] * 3, img.shape[1] * 3),
    )
    write_image(distorted_img, "images/test_distorted_img.png")
