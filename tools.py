"""
Author: Ricardo Grande Cros
Created: 03/01/2022

This class contains useful methods
"""


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
