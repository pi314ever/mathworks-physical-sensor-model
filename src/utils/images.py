# Util for generating and manipulating image formats

# Image formats are in BGR for convenience to work with cv2 library

import cv2

def read_image(filename: str):
    """
    Wrapper around cv2.imread(). Reads an image from a file

    Args:
        filename (str): Path to image
    """
    raw_img = cv2.imread(filename)
    return raw_img
    return cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)