# File for path constants and helper functions

import os

ROOT_PATH = os.path.dirname(os.path.abspath(os.path.join(__file__,'../../')))

DATA_PATH = os.path.join(ROOT_PATH, 'data')

SOURCE_PATH = os.path.join(ROOT_PATH, 'src')

def get_data_path(*args) -> str:
    """
    Wrapper around os join for data path directory

    Returns:
        str: path to a file in the data directory
    """
    return os.path.join(DATA_PATH, *args)

def find_data(*args):
    """
    Finds the path to a file in the data directory

    Args:
        *args: path to file in data directory

    Returns:
        str: path to the file
    """
    for root, _, files in os.walk(get_data_path(*args)):
        for file in files:
            yield os.path.join(root, file)
