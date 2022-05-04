#
# parakeet.io.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import functools
import h5py
import numpy as np
import mrcfile
import os
import PIL.Image
import parakeet

FEI_EXTENDED_HEADER_DTYPE = mrcfile.dtypes.get_ext_header_dtype(b"FEI1")

# metadata_dtype = np.dtype([
#    #
#    # General parameters
#    #
#    ("application", 'S16'),
#    ("application_version", 'S16'),
#    ("timestamp", 'f8'),
#    #
#    # Stage parameters
#    #
#    ('tilt_alpha', 'f8'),
#    ('tilt_axis_angle', 'f8'),
#    ('stage_x', 'f8'),
#    ('stage_y', 'f8'),
#    ('stage_z', 'f8'),
#    #
#    # Beam parameters
#    #
#    ('energy', 'f8'),
#    ('dose', 'f8'),
#    ('slit_inserted', '?'),
#    ('slit_width', 'f8'),
#    ('energy_shift', 'f8'),
#    ('shift_x', 'f8'),
#    ('shift_y', 'f8'),
#    ('shift_offset_x', 'f8'),
#    ('shift_offset_y', 'f8'),
#    ('acceleration_voltage_spread', 'f8'),
#    ('energy_spread', 'f8'),
#    ('source_spread', 'f8'),
#    ('exposure_time', 'f8'),
#    ('theta', 'f8'),
#    ('phi', 'f8'),
#    #
#    # Detector parameters
#    #
#    ('pixel_size_x', 'f8'),
#    ('pixel_size_y', 'f8'),
#    ('image_size_x', 'i4'),
#    ('image_size_y', 'i4'),
#    ('gain', 'f8'),
#    ('offset', 'f8'),
#    ('dqe', '?'),
#    #
#    # Lens parameters
#    #
#    ("c_10", 'f8'),
#    ("c_12", 'f8'),
#    ("c_21", 'f8'),
#    ("c_23", 'f8'),
#    ("c_30", 'f8'),
#    ("c_32", 'f8'),
#    ("c_34", 'f8'),
#    ("c_41", 'f8'),
#    ("c_43", 'f8'),
#    ("c_45", 'f8'),
#    ("c_50", 'f8'),
#    ("c_52", 'f8'),
#    ("c_54", 'f8'),
#    ("c_56", 'f8'),
#    ("c_c", 'f8'),
#    ("phi_12", 'f8'),
#    ("phi_21", 'f8'),
#    ("phi_23", 'f8'),
#    ("phi_32", 'f8'),
#    ("phi_34", 'f8'),
#    ("phi_41", 'f8'),
#    ("phi_43", 'f8'),
#    ("phi_45", 'f8'),
#    ("phi_52", 'f8'),
#    ("phi_54", 'f8'),
#    ("phi_56", 'f8'),
#    ("current_spread", 'f8'),
#    ('phase_plate', '?'),
#    #
#    # Simulation parameters
#    #
#    ('slice_thickness', 'f8'),
#    ('ice', '?'),
#    ('inelastic_model', 'S16'),
#    ('damage_model', '?'),
#    ('sensitivity_coefficient', 'f8'),
# ])

METADATA_DTYPE = np.dtype(
    [
        #
        # General parameters
        #
        ("application", "S16"),
        ("application_version", "S16"),
        ("timestamp", "f8"),
        #
        # Stage parameters
        #
        ("tilt_alpha", "f8"),
        ("stage_z", "f8"),
        #
        # Beam parameters
        #
        ("shift_x", "f8"),
        ("shift_y", "f8"),
        ("shift_offset_x", "f8"),
        ("shift_offset_y", "f8"),
        #
        # Detector parameters
        #
        ("pixel_size_x", "f8"),
        ("pixel_size_y", "f8"),
        #
        # Lens parameters
        #
        ("c_10", "f8"),
    ]
)


class Row(object):
    def __init__(self, header, index: int):
        self._header = header
        self._index = index

    @property
    def size(self):
        return np.arange(self._header.size)[self._index].size

    def indices(self, item):
        return np.arange(self.size)[item]

    def __getitem__(self, key):
        return self._header.get(self._index, key)

    def __setitem__(self, key, value):
        self._header.set(self._index, key, value)

    def assign(self, value):
        for key in self._header.dtype.fields:
            self[key] = value[key]

    def __array__(self, dtype=METADATA_DTYPE):
        result = np.zeros(self.size, dtype=self._header.dtype)
        for key in self._header.dtype.fields:
            result[key] = self[key]
        return result.astype(dtype)


class Column(object):
    def __init__(self, header, key: str):
        self._header = header
        self._key = key

    @property
    def size(self):
        return 1

    def __getitem__(self, index):
        return self._header.get(index, self._key)

    def __setitem__(self, index, value):
        self._header.set(index, self._key, value)

    def __array__(self, dtype=None):
        return self[:]


class Header(object):
    @functools.singledispatchmethod
    def __getitem__(self, item):
        return Row(self, item)

    @__getitem__.register
    def _(self, item: str):
        return Column(self, item)

    @functools.singledispatchmethod
    def __setitem__(self, item, value):
        row = Row(self, item)
        row.assign(value)

    @__setitem__.register
    def _(self, item: str, value):
        col = Column(self, item)
        col[:] = value

    def rows(self):
        for i in range(self.size):
            yield Row(self, i)

    def cols(self):
        for key in self.fields:
            yield Column(self, key)

    @property
    def fields(self):
        return self.dtype.fields

    @property
    def dtype(self):
        return METADATA_DTYPE

    @property
    def position(self):
        result = np.zeros(shape=(self.size, 3), dtype=np.float32)
        result[:, 0] = self["shift_x"][:]
        result[:, 1] = self["shift_y"][:]
        result[:, 2] = self["stage_z"][:]
        return result

    def __array__(self, dtype=METADATA_DTYPE):
        return np.asarray(self[:])


class Writer(object):
    """
    Interface to write the simulated data

    """

    @property
    def shape(self):
        """
        The shape property

        """
        return self._data.shape

    @property
    def dtype(self):
        """
        The dtype property

        """
        return self._data.dtype

    @property
    def data(self):
        """
        The data property

        """
        return self._data

    @property
    def header(self):
        """
        The header metdata

        """
        return self._header

    @property
    def is_mrcfile_writer(self):
        """
        Return if is mrcfile

        """
        return isinstance(self, MrcFileWriter)

    @property
    def is_nexus_writer(self):
        """
        Return if is nexus

        """
        return isinstance(self, NexusWriter)

    @property
    def is_image_writer(self):
        """
        Return if is image

        """
        return isinstance(self, ImageWriter)

    def update(self):
        """
        Update if anything needs to be done

        """
        pass


class MrcfileHeader(Header):
    def __init__(self, handle):
        self._handle = handle

    @classmethod
    def mapping(Class, key):
        return {
            "application": "Application",
            "application_version": "Application version",
            "timestamp": "Timestamp",
            "tilt_alpha": "Alpha tilt",
            "stage_z": "Z-Stage",
            "shift_x": "Shift X",
            "shift_y": "Shift Y",
            "shift_offset_x": "Shift offset X",
            "shift_offset_y": "Shift offset Y",
            "pixel_size_x": "Pixel size X",
            "pixel_size_y": "Pixel size X",
            "c_10": "Defocus",
        }[key]

    def get(self, index, key):
        mapping = self.mapping(key)
        getter = {
            "pixel_size_x": lambda x: x * 1e-10,
            "pixel_size_y": lambda x: x * 1e-10,
        }.get(key, lambda x: x)
        return getter(self._handle[index][mapping])

    def set(self, index, key, value):
        mapping = self.mapping(key)
        setter = {
            "pixel_size_x": lambda x: x * 1e10,
            "pixel_size_y": lambda x: x * 1e10,
        }.get(key, lambda x: x)
        self._handle[index][mapping] = setter(value)

    @property
    def size(self):
        return self._handle.shape[0]


class MrcFileWriter(Writer):
    """
    Write to an mrcfile

    """

    def __init__(self, filename, shape, pixel_size, dtype="uint8"):
        """
        Initialise the writer

        Args:
            filename (str): The filename
            shape (tuple): The shape of the data
            pixel_size (float): The pixel size
            dtype (str): The data type

        """
        # Get the dtype
        dtype = np.dtype(dtype)

        # Convert 32bit int and 64bit float
        if dtype == "int32":
            dtype = np.dtype(np.int16)
        elif dtype == "uint32":
            dtype = np.dtype(np.uint16)
        elif dtype == "float64":
            dtype = np.dtype(np.float32)
        elif dtype == "complex128":
            dtype = np.dtype(np.complex64)

        # Open the handle to the mrcfile
        self.handle = mrcfile.new_mmap(
            filename,
            shape=shape,
            mrc_mode=mrcfile.utils.mode_from_dtype(dtype),
            overwrite=True,
            extended_header=np.zeros(shape=shape[0], dtype=FEI_EXTENDED_HEADER_DTYPE),
            exttyp="FEI1",
        )

        # Set the data array
        self._data = self.handle.data

        # Create the header info
        self._header = MrcfileHeader(self.handle.extended_header)

        # Set the pixel size
        self.handle.voxel_size = pixel_size
        self._header[:]["pixel_size_x"] = pixel_size
        self._header[:]["pixel_size_y"] = pixel_size
        self._header[:]["application"] = "Parakeet"
        self._header[:]["application_version"] = parakeet.__version__

    @property
    def pixel_size(self):
        """
        The pixel size

        """
        return self.handle.voxel_size[0]

    def update(self):
        """
        Update before closing

        """
        self.handle.update_header_stats()


class NexusHeader(Header):
    def __init__(self, handle):
        self._handle = handle

    @classmethod
    def mapping(Class, key):
        return {
            "application": "application",
            "application_version": "application_version",
            "timestamp": "timestamp",
            "tilt_alpha": "rotation_angle",
            "stage_z": "z_translation",
            "shift_x": "x_translation",
            "shift_y": "y_translation",
            "shift_offset_x": "x_drift",
            "shift_offset_y": "y_drift",
            "pixel_size_x": "x_pixel_size",
            "pixel_size_y": "y_pixel_size",
            "c_10": "defocus",
        }[key]

    def get(self, index, key):
        mapping = self.mapping(key)
        return self._handle[mapping][index]

    def set(self, index, key, value):
        mapping = self.mapping(key)
        self._handle[mapping][index] = value

    @property
    def size(self):
        return self._handle["data"].shape[0]


class NexusWriter(Writer):
    """
    Write to a nexus file

    """

    def __init__(self, filename, shape, pixel_size, dtype="float32"):
        """
        Initialise the writer

        Args:
            filename (str): The filename
            shape (tuple): The shape of the data
            pixel_size (float): The pixel size
            dtype (object): The data type of the data

        """

        # Open the file for writing
        self.handle = h5py.File(filename, "w")

        # Create the entry
        entry = self.handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry["definition"] = "NXtomo"

        # Create the instrument
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        # Create the detector
        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("data", shape=shape, dtype=dtype)
        detector["image_key"] = np.zeros(shape=shape[0])
        detector["x_pixel_size"] = np.full(shape=shape[0], fill_value=pixel_size)
        detector["y_pixel_size"] = np.full(shape=shape[0], fill_value=pixel_size)
        detector.create_dataset("timestamp", shape=(shape[0],), dtype=np.float64)

        # Create the sample
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample["name"] = "parakeet-simulation"
        sample.create_dataset("application", shape=(shape[0],), dtype="S16")
        sample.create_dataset("application_version", shape=(shape[0],), dtype="S16")
        sample.create_dataset("rotation_angle", shape=(shape[0],), dtype=np.float32)
        sample.create_dataset("x_translation", shape=(shape[0],), dtype=np.float32)
        sample.create_dataset("y_translation", shape=(shape[0],), dtype=np.float32)
        sample.create_dataset("z_translation", shape=(shape[0],), dtype=np.float32)
        sample.create_dataset("x_drift", shape=(shape[0],), dtype=np.float32)
        sample.create_dataset("y_drift", shape=(shape[0],), dtype=np.float32)
        sample.create_dataset("defocus", shape=(shape[0],), dtype=np.float32)

        # Create the data
        data = entry.create_group("data")
        data["data"] = detector["data"]
        data["application"] = sample["application"]
        data["application_version"] = sample["application_version"]
        data["rotation_angle"] = sample["rotation_angle"]
        data["x_translation"] = sample["x_translation"]
        data["y_translation"] = sample["y_translation"]
        data["z_translation"] = sample["z_translation"]
        data["x_drift"] = sample["x_drift"]
        data["y_drift"] = sample["y_drift"]
        data["defocus"] = sample["defocus"]
        data["image_key"] = detector["image_key"]
        data["timestamp"] = detector["timestamp"]
        data["x_pixel_size"] = detector["x_pixel_size"]
        data["y_pixel_size"] = detector["y_pixel_size"]

        # Set the data ptr
        self._data = data["data"]
        self._header = NexusHeader(data)
        self._header[:]["pixel_size_x"] = pixel_size
        self._header[:]["pixel_size_y"] = pixel_size
        self._header[:]["application"] = "Parakeet"
        self._header[:]["application_version"] = parakeet.__version__

    @property
    def pixel_size(self):
        """
        Return the pixel size

        """
        return self.handle["instrument"]["detector"]["x_pixel_size"][0]


class ImageWriter(Writer):
    """
    Write to a images

    """

    class DataProxy(object):
        """
        A proxy interface for the data

        """

        def __init__(self, template, shape=None, vmin=None, vmax=None):
            self.template = template
            self.shape = shape
            self.vmin = vmin
            self.vmax = vmax

        def __setitem__(self, item, data):

            # Check the input
            assert isinstance(item, tuple)
            assert isinstance(item[1], slice)
            assert isinstance(item[2], slice)
            assert item[1].start is None
            assert item[1].stop is None
            assert item[1].step is None
            assert item[2].start is None
            assert item[2].stop is None
            assert item[2].step is None
            assert len(data.shape) == 2
            assert data.shape[0] == self.shape[1]
            assert data.shape[1] == self.shape[2]

            # Convert to squared amplitude
            if np.iscomplexobj(data):
                data = np.abs(data) ** 2
                self.vmin = None
                self.vmax = None

            # Compute scale factors to put between 0 and 255
            if self.vmin is None:
                vmin = np.min(data)
            else:
                vmin = self.vmin
            if self.vmax is None:
                vmax = np.max(data)
            else:
                vmax = self.vmax
            s1 = 255.0 / (vmax - vmin)
            s0 = -s1 * vmin

            # Save the image to file
            filename = self.template % (item[0] + 1)
            image = (data * s1 + s0).astype(np.uint8)
            PIL.Image.fromarray(image).save(filename)

    def __init__(self, template, shape=None, vmin=None, vmax=None):
        """
        Initialise the writer

        Args:
            filename (str): The filename
            shape (tuple): The shape of the data

        """

        # Set the proxy data interface
        self._data = ImageWriter.DataProxy(template, shape, vmin, vmax)

        # Create dummy arrays for angle and position
        self._header = np.zeros(shape=shape[0], dtype=METADATA_DTYPE)

    @property
    def vmin(self):
        """
        The vmin property

        """
        return self._data.vmin

    @vmin.setter
    def vmin(self, vmin):
        """
        The vmin propety setter

        """
        self._data.vmin = vmin

    @property
    def vmax(self):
        """
        The vmax property

        """
        return self._data.vmax

    @vmax.setter
    def vmax(self, vmax):
        """
        The vmax property setter

        """
        self._data.vmax = vmax


class Reader(object):
    """
    Interface to write the simulated data

    """

    def __init__(
        self,
        handle,
        data,
        header,
        pixel_size,
    ):
        """
        Initialise the data

        Args:
            data (array): The data array
            header (array): The header data

        """
        # Check the size
        assert header.size == data.shape[0], "Inconsistent dimensions"

        # Set the array
        self.handle = handle
        self.data = data
        self.header = header
        self.pixel_size = pixel_size
        self.shape = data.shape
        self.dtype = data.dtype

    @property
    def start_angle(self):
        """
        Returns:
            float: the start angle

        """
        return self.header[-1]["tilt_alpha"]

    @property
    def stop_angle(self):
        """
        Returns:
            float: The stop angle

        """
        return self.header[-1]["tilt_alpha"]

    @property
    def step_angle(self):
        """
        Returns:
            float: The stop angle

        """
        # tol = 1e-7
        if len(self.header) == 1:
            step = 1
        else:
            step = (self.stop_angle - self.start_angle) / (self.num_images - 1)
        return step

    @property
    def num_images(self):
        """
        Returns:
            int: The number of images

        """
        return len(self.header)

    @classmethod
    def from_mrcfile(Class, filename):
        """
        Read the simulated data from a mrc file

        Args:
            filename (str): The input filename

        """

        # Read the data
        handle = mrcfile.mmap(filename, "r")

        # Check the header info
        if handle.header.exttyp == b"FEI1":
            assert handle.extended_header.dtype == FEI_EXTENDED_HEADER_DTYPE
            assert len(handle.extended_header.shape) == 1
            assert handle.extended_header.shape[0] == handle.data.shape[0]
            header = MrcfileHeader(handle.extended_header)
        else:
            extended_header = np.zeros(
                shape=handle.data.shape[0], dtype=FEI_EXTENDED_HEADER_DTYPE
            )
            header = MrcfileHeader(handle.extended_header)

        # Get the pixel size
        pixel_size = handle.voxel_size["x"]

        # Create the reader
        return Reader(
            handle,
            handle.data,
            header,
            pixel_size,
        )

    @classmethod
    def from_nexus(Class, filename):
        """
        Read the simulated data from a nexus file

        Args:
            filename (str): The input filename

        """

        # Read the data from disk
        handle = h5py.File(filename, "r")

        # Get the entry
        entry = handle["entry"]
        assert entry.attrs["NX_class"] == "NXentry"
        definition = entry["definition"][()]
        if isinstance(definition, bytes):
            definition = definition.decode("utf-8")
        assert definition == "NXtomo"

        # Get the data and detector
        data = entry["data"]
        detector = entry["instrument"]["detector"]

        # Get the header
        header = NexusHeader(data)

        # Get the pixel size
        pixel_size = detector["x_pixel_size"][0]

        # Create the reader
        return Reader(
            handle,
            data["data"],
            header,
            pixel_size,
        )

    @classmethod
    def from_file(Class, filename):
        """
        Read the simulated data from file

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


def new(filename, shape=None, pixel_size=1, dtype="float32", vmin=None, vmax=None):
    """
    Create a new file for writing

    Args:
        filename (str): The output filename
        shape (tuple): The output shape
        pixel_size (tuple): The pixel size
        dtype (object): The data type (only used with NexusWriter)
        vmin (int): The minimum value (only used in ImageWriter)
        vmax (int): The maximum value (only used in ImageWriter)

    Returns:
        object: The file writer

    """
    extension = os.path.splitext(filename)[1].lower()
    if extension in [".mrc"]:
        return MrcFileWriter(filename, shape, pixel_size, dtype)
    elif extension in [".h5", ".hdf5", ".nx", ".nxs", ".nexus", "nxtomo"]:
        return NexusWriter(filename, shape, pixel_size, dtype)
    elif extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        return ImageWriter(filename, shape, vmin, vmax)
    else:
        raise RuntimeError(f"File with unknown extension: {filename}")


def open(filename):
    """
    Read the simulated data from file

    Args:
        filename (str): The output filename

    """
    return Reader.from_file(filename)
