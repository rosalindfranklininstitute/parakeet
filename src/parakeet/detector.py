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
import parakeet.config
from typing import Tuple


class Detector(object):
    """
    A class to encapsulate a detector

    """

    def __init__(
        self,
        nx: int = 0,
        ny: int = 0,
        pixel_size: float = 1,
        origin: Tuple[float, float, float] = (0, 0, 0),
        dqe: bool = True,
    ):
        """
        Initialise the detector

        Args:
            nx: The size of the detector in X
            ny: The size of the detector in Y
            pixel_size: The effective size of the pixels in A
            origin: The detector origin (A, A)
            dqe: Use the detector DQE

        """
        self.nx = nx
        self.ny = ny
        self.pixel_size = pixel_size
        self.origin = origin
        self.dqe = dqe


def new(config: parakeet.config.Detector) -> Detector:
    """
    Make a new detector

    Args:
        config: The detector configuration

    Returns:
        The detector model object

    """
    return Detector(**config.model_dump())
