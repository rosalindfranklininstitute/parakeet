#
# parakeet.data.__init__.py
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
    return os.path.join(os.path.dirname(__file__), "files", name)


def get_pdb(name):
    """
    Return the path to the data file

    Args:
        name (str): The pdb id

    Returns:
        str: The absolute filename

    """

    def get_filenames_to_check(name):
        return [get_path("%s.%s" % (name, ext)) for ext in ["pdb", "cif"]]

    def get_valid_filenames(name):
        return [
            filename
            for filename in get_filenames_to_check(name)
            if os.path.exists(filename)
        ]

    return get_valid_filenames(name)[0]


def get_4v1w():
    """
    Get the path to the file containing the structure of Apoferratin (4v1w)

    Returns:
        str: The abolsute path to 4v1w

    """
    return get_path("4v1w.cif")


def get_4v5d():
    """
    Get the path to the file containing the structure of ribosome (4v5d)

    Returns:
        str: The abolsute path to 4v5d

    """
    return get_path("4v5d.cif")


def get_6qt9():
    """
    Get the path to the file containing the structure of the icosohedral virus
    SH1 (6qt9)

    Returns:
        str: The abolsute path to 6qt9

    """
    return get_path("6qt9.cif")
