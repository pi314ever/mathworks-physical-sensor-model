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


def get_xy_range(resolution):
    scale = max(resolution)
    x_min, x_max = -resolution[0] / scale, resolution[0] / scale
    y_min, y_max = -resolution[1] / scale, resolution[1] / scale
    x_range = np.linspace(x_min, x_max, resolution[0])
    y_range = np.linspace(y_min, y_max, resolution[1])
    return x_range, y_range


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
        x_range, y_range = get_xy_range(output_resolution)
        x_range_input, y_range_input = get_xy_range(images.shape[:2])

        # Run output point maps through neural network to get query points
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        x_mesh = x_mesh.flatten()
        y_mesh = y_mesh.flatten()
        vstack = np.vstack((x_mesh, y_mesh))
        vstack = vstack.T

        K_np = np.array(K, dtype=np.float64)
        P_np = np.array(P, dtype=np.float64)

        print("Running neural network")

        query_points = self.point_map_model.predict(vstack, K_np, P_np)
        print("Interpolating")
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

    from model import CombinedPMNN, SeparatePMNN, load_model_from_dirs

    parser = ArgumentParser()
    parser.add_argument("model_dirs", nargs="+", type=str)
    parser.add_argument(
        "--image_path", type=str, default=get_data_path("images", "checkerboard.jpg")
    )
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    img = read_image(args.image_path)
    model = load_model_from_dirs(args.model_dirs)

    pmmp = PointMapModelPipeline(model, "nearest")
    distorted_img = pmmp(
        img,
        (0.01, 0.03, 0.02),
        (0.00, 0.00),
        output_resolution=(img.shape[0] * 3, img.shape[1] * 3),
    )
    write_image(distorted_img, "images/test_distorted_img.png")
