#
# parakeet.__init__.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


try:
    from parakeet._version import version as __version__
except ImportError:
    __version__ = "unknown"


from parakeet._run import run  # noqa
