# Pipeline class that encapsulates the entire pipeline for the point map nn

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import tensorflow as tf
from typing import Optional, Union
from numpy.typing import NDArray
from utils.images import read_image
from utils._paths import get_data_path

class PointMapModelPipeline:
    """
    Point map neural network pipeline class

    Given initial setup (trained neural network model, interpolation model),
    """
    point_map_model: tf.keras.models.Sequential


    def __init__(self, point_map_model, interpolation_model) -> None:
        self.point_map_model
        pass

    def __call__(self, images: Union[list[NDArray], NDArray], output_resolution: Optional[tuple[int, int]] = None):
        """
        Takes in either a single picture or a batch of pictures with output resolutions and returns the corresponding distorted pictures
        """
        if isinstance(images, list):
            raise NotImplementedError

        # Convert images to point maps

        # Get output resolution point map locations

        # Run output point maps through neural network to get query points

        # Interpolate on all query points

        # Convert output point map to images

    def _image_to_point_maps(self, image: NDArray):
        """
        Image comes in as a NDArray of shape (height, width, 3) with values in range [0, 255]. Processes the images into point map data objects (with normalized pixel locations and )
        """
        pass

class PointMapData:
    """
    Class for storing point location (integer pixel locations and floating point normalized locations) and corresponding pixel values for a single image
    """

    image_raw: NDArray                      # Raw image data
    resolution: tuple[int, int]             # Image resolution
    pixel_idx2normed_coordinates: dict[tuple[int, int], tuple[float, float]]  # Mapping from pixel indices to normalized coordinates
    pixel_values: NDArray                   # Pixel values for each pixel in the image, accessed by pixel_values[y, x] -> tuple[int, int, int]

    def __init__(self) -> None:
        pass

if __name__ == '__main__':
    img = read_image(get_data_path('images','stanford.png'))
    print(img)

    point_map_data = PointMapData()