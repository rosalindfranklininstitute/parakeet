#
# elfantasma.scan.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy


class Scan(object):
    """
    A class to encapsulate an scan

    """

    def __init__(self, axis=None, angles=None, positions=None):
        """
        Initialise the scan

        Args:
            axis (tuple): The rotation axis
            angles (list): The rotation angles (units: degrees)
            positions (list): The positions to shift(units: A)

        """
        if axis is None:
            self.axis = (1, 0, 0)
        else:
            self.axis = axis
        if angles is None:
            self.angles = [0]
        else:
            self.angles = angles
        if positions is None:
            self.positions = [0]
        else:
            self.positions = positions
        assert len(self.angles) == len(self.positions)


def create_scan(
    mode="still",
    axis=(1, 0, 0),
    start_angle=0,
    stop_angle=0,
    step_angle=0,
    start_pos=0,
    stop_pos=0,
    step_pos=0,
):
    """
    Create an scan

    """
    if mode == "still":
        return Scan(axis=axis, angles=[start_angle], positions=[start_pos])
    elif mode == "tilt_series":
        return Scan(axis=axis, angles=numpy.arange(start_angle, stop_angle, step_angle))
    elif mode == "helical_scan":
        angles = numpy.arange(start_angle, stop_angle, step_angle)
        step_pos = (stop_pos - start_pos) / len(angles)
        positions = numpy.arange(start_pos, stop_pos, step_pos)
        return Scan(axis=axis, angles=angles, positions=positions)
