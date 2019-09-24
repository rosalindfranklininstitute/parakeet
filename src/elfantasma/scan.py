#
# elfantasma.scan.py
#
# Copyright (C) 2019 Diamond Light Source
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


class Scan(object):
    """
    A class to encapsulate an scan

    """

    def __init__(self, axis, angles):
        """
        Initialise the scan

        Args:
            axis (tuple): The rotation axis
            angles (list): The rotation angles (units: degrees)

        """
        self.axis = axis
        self.angles = angles


def create_scan(
    mode="still", axis=(1, 0, 0), start_angle=0, stop_angle=0, step_angle=0
):
    """
    Create an scan

    """
    if mode == "still":
        return Scan(axis=(1, 0, 0), angles=[0])
    elif mode == "tilt_series":
        return Scan(axis=axis, angles=numpy.arange(start_angle, stop_angle, step_angle))
