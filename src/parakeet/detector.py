#
# parakeet.detector.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


class Detector(object):
    """
    A class to encapsulate a detector

    """

    def __init__(self, nx=None, ny=None, pixel_size=None, origin=None, dqe=True):
        """
        Initialise the detector

        Args:
            nx (int): The size of the detector in X
            ny (int): The size of the detector in Y
            pixel_size (float): The effective size of the pixels in A
            origin (tuple): The detector origin (A, A)
            dqe (bool): Use the detector DQE

        """
        self.nx = nx
        self.ny = ny
        self.pixel_size = pixel_size
        self.origin = origin
        self.dqe = dqe


def new(nx=None, ny=None, pixel_size=None, origin=None, dqe=True):
    """
    Make a new detector

    Args:
        nx (int): The size of the detector in X
        ny (int): The size of the detector in Y
        pixel_size (float): The effective size of the pixels in A
        origin (tuple): The detector origin (A, A)
        dqe (bool): Use the detector DQE

    Returns:
        obj: The detector object

    """
    return Detector(nx=nx, ny=ny, pixel_size=pixel_size, origin=origin, dqe=dqe)
