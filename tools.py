"""
Author: Ricardo Grande Cros
Created: 03/01/2022

This class contains useful methods
"""
import numpy as np


def read_file(file):
    """
    Reads a TSP file in tsplib95 style
    https://pypi.org/project/tsplib95/
    :param file:
    :return:
    """
    with open(file, 'r') as reader:
        content = reader.read().splitlines()
        lines = [x.lstrip() for x in content if x != ""]
        return lines


def normalize_data(vector: np.ndarray):
    vector[:, 0] = (vector[:, 0] - vector[:, 0].min()) / (vector[:, 0].max() - vector[:, 0].min())
    vector[:, 1] = (vector[:, 1] - vector[:, 1].min()) / (vector[:, 1].max() - vector[:, 1].min())

    return vector
