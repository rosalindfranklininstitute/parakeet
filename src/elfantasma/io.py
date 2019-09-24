#
# elfantasma.io.py
#
# Copyright (C) 2019 Diamond Light Source
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import h5py
import numpy
import mrcfile
import os
import PIL.Image


class Writer(object):
    """
    Interface to write the simulated data

    """

    def __init__(self, shape=None):
        """
        Initialise the data

        Args:
            shape (tuple): The dimensions of the data

        """
        if shape is None:
            shape = (0, 0, 0)

        # Instantiate
        self._data = numpy.zeros(shape=shape, dtype=numpy.float32)
        self._angle = numpy.zeros(shape=shape[0], dtype=numpy.float32)

    @property
    def shape(self):
        """
        The shape property

        """
        return self._data.shape

    @shape.setter
    def shape(self, s):
        """
        Set the shape property

        """
        self._data.resize(s)
        self._angle.resize(s[0])

    @property
    def data(self):
        """
        The data property

        """
        return self._data

    @property
    def angle(self):
        """
        The angle property

        """
        return self._angle

    def as_mrcfile(self, filename):
        """
        Write the simulated data to a mrc file

        Args:
            filename (str): The output filename

        """
        assert len(self.angle) == self.data.shape[0], "Inconsistent dimensions"

        # Write the data
        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(self.data)

        # Write the rotation angles
        with open(f"{filename}.angles", "w") as outfile:
            for angle in self.angle:
                outfile.write("%.2f\n" % angle)

    def as_nexus(self, filename):
        """
        Write the simulated data to a nexus file

        Args:
            filename (str): The output filename

        """

        # Ensure that the angles matches the data
        assert len(self.angle) == self.data.shape[0], "Inconsistent dimensions"

        # Write the data to disk
        with h5py.File(filename, "w") as outfile:

            # Create the entry
            entry = outfile.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            entry["definition"] = "NXtomo"

            # Create the instrument
            instrument = entry.create_group("instrument")
            instrument.attrs["NX_class"] = "NXinstrument"

            # Create the detector
            detector = instrument.create_group("detector")
            detector.attrs["NX_class"] = "NXdetector"
            detector["data"] = self.data
            detector["image_key"] = numpy.zeros(shape=(len(self.angle),))

            # Create the sample
            sample = entry.create_group("sample")
            sample.attrs["NX_class"] = "NXsample"
            sample["name"] = "elfantasma-simulation"
            sample["rotation_angle"] = self.angle

            # Create the data
            data = entry.create_group("data")
            data["data"] = detector["data"]
            data["rotation_angle"] = sample["rotation_angle"]
            data["image_key"] = detector["image_key"]

    def as_images(self, template):
        """
        Write the simulated data as a sequence of png files

        Args:
            template (str): The output template (e.g. "image_%03d.png")

        """
        # Compute scale factors to put between 0 and 255
        min_value = numpy.min(self.data)
        max_value = numpy.max(self.data)
        s1 = 255.0 / (max_value - min_value)
        s0 = -s1 * min_value

        # Write each image to a PNG
        for i in range(self.data.shape[0]):
            image = self.data[i, :, :] * s1 + s0
            image = image.astype(numpy.uint8)
            filename = template % (i + 1)
            print(f"    writing image {i+1} to {filename}")
            PIL.Image.fromarray(image).save(filename)

    def as_file(self, filename):
        """
        Write the simulated data to file

        Args:
            filename (str): The output filename

        """
        extension = os.path.splitext(filename)[1].lower()
        if extension in [".mrc"]:
            self.as_mrcfile(filename)
        elif extension in [".h5", ".hdf5", ".nx", ".nxs", ".nexus", "nxtomo"]:
            self.as_nexus(filename)
        elif extension in [".png", ".jpg", ".jpeg"]:
            self.as_images(filename)
        else:
            raise RuntimeError(f"File with unknown extension: {filename}")


class Reader(object):
    """
    Interface to write the simulated data

    """

    def __init__(self, data, angle):
        """
        Initialise the data

        Args:
            data (array): The data array
            angle (array): The angle array

        """
        # Check the size
        assert len(angle) == data.shape[0], "Inconsistent dimensions"

        # Set the array
        self.data = data
        self.angle = angle

    @classmethod
    def from_mrcfile(Class, filename):
        """
        Read the simulated data from a mrc file

        Args:
            filename (str): The input filename

        """

        # Read the data
        with mrcfile.open(filename) as mrc:
            data = mrc.data

        # Read the rotation angles
        with open(f"{filename}.angles", "r") as infile:
            angle = numpy.array(list(infile.readlines()))

        # Create the reader
        return Reader(data, angle)

    @classmethod
    def from_nexus(Class, filename):
        """
        Read the simulated data from a nexus file

        Args:
            filename (str): The input filename

        """

        # Read the data from disk
        with h5py.File(filename, "r") as infile:

            # Get the entry
            entry = infile["entry"]
            assert entry.attrs["NX_class"] == "NXentry"
            assert entry["definition"][()] == "NXtomo"

            # Get the data
            data = entry["data"]

            # Create the reader
            return Reader(data["data"][:], data["rotation_angle"][:])

    @classmethod
    def from_file(Class, filename):
        """
        Write the simulated data to file

        Args:
            filename (str): The output filename

        """
        extension = os.path.splitext(filename)[1].lower()
        if extension in [".mrc"]:
            return Class.from_mrcfile(filename)
        elif extension in [".h5", ".hdf5", ".nx", ".nxs", ".nexus", "nxtomo"]:
            return Class.from_nexus(filename)
        else:
            raise RuntimeError(f"File with unknown extension: {filename}")
