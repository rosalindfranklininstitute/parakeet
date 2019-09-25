#
# elfantasma.data.__init__.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import os.path


def get_path(name):
    """
    Return the path to the data file

    Args:
        name (str): The relative filename

    Returns:
        str: The absolute filename

    """
    return os.path.join(os.path.dirname(__file__), name)
