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
import h5py
import numpy as np
import mrcfile
import os
import PIL.Image
import parakeet

# try:
#     FEI_EXTENDED_HEADER_DTYPE = mrcfile.dtypes.FEI1_EXTENDED_HEADER_DTYPE
# except Exception:
FEI_EXTENDED_HEADER_DTYPE = mrcfile.dtypes.get_ext_header_dtype(b"FEI1")

METADATA_DTYPE = np.dtype(
    [
        #
        # General parameters
        #
        ("application", "S16"),
        ("application_version", "S16"),
        ("timestamp", "f8"),
        #
        # Scan parameters
        #
        ("image_number", "i4"),
        ("fraction_number", "i4"),
        #
        # Stage parameters
        #
        ("tilt_alpha", "f8"),
        ("tilt_axis_x", "f8"),
        ("tilt_axis_y", "f8"),
        ("tilt_axis_z", "f8"),
        ("stage_x", "f8"),
        ("stage_y", "f8"),
        ("stage_z", "f8"),
        ("stage_offset_x", "f8"),
        ("stage_offset_y", "f8"),
        ("stage_offset_z", "f8"),
        #
        # Beam parameters
        #
        ("energy", "f8"),
        ("dose", "f8"),
        ("slit_inserted", "?"),
        ("slit_width", "f8"),
        ("energy_shift", "f8"),
        ("acceleration_voltage_spread", "f8"),
        ("energy_spread", "f8"),
        ("illumination_semiangle", "f8"),
        ("exposure_time", "f8"),
        ("theta", "f8"),
        ("phi", "f8"),
        ("shift_x", "f8"),
        ("shift_y", "f8"),
        ("shift_offset_x", "f8"),
        ("shift_offset_y", "f8"),
        ("shift_offset_z", "f8"),
        #
        # Detector parameters
        #
        ("pixel_size_x", "f8"),
        ("pixel_size_y", "f8"),
        ("image_size_x", "i4"),
        ("image_size_y", "i4"),
        ("gain", "f8"),
        ("offset", "f8"),
        ("dqe", "?"),
        #
        # Lens parameters
        #
        ("c_10", "f8"),
        ("c_12", "f8"),
        ("c_21", "f8"),
        ("c_23", "f8"),
        ("c_30", "f8"),
        ("c_32", "f8"),
        ("c_34", "f8"),
        ("c_41", "f8"),
        ("c_43", "f8"),
        ("c_45", "f8"),
        ("c_50", "f8"),
        ("c_52", "f8"),
        ("c_54", "f8"),
        ("c_56", "f8"),
        ("phi_12", "f8"),
        ("phi_21", "f8"),
        ("phi_23", "f8"),
        ("phi_32", "f8"),
        ("phi_34", "f8"),
        ("phi_41", "f8"),
        ("phi_43", "f8"),
        ("phi_45", "f8"),
        ("phi_52", "f8"),
        ("phi_54", "f8"),
        ("phi_56", "f8"),
        ("c_c", "f8"),
        ("current_spread", "f8"),
        ("phase_plate", "?"),
        #
        # Simulation parameters
        #
        ("slice_thickness", "f8"),
        ("ice", "?"),
        ("inelastic_model", "S16"),
        ("damage_model", "?"),
        ("sensitivity_coefficient", "f8"),
    ]
)


class Row(object):
    """
    An object to represent a row

    """

    def __init__(self, header, index):
        self._header = header
        self._index = index

    @property
    def size(self) -> int:
        """
        The size of the header

        """
        return np.arange(self._header.size)[self._index].size

    def indices(self, item) -> np.ndarray:
        """
        Args:
            item: The index

        Returns:
            The row indices

        """
        return np.arange(self.size)[item]

    def __getitem__(self, key: str):
        """
        Get an item from the row

        Args:
            key: The field name

        Returns:
            The row element

        """
        return self._header.get(self._index, key)

    def __setitem__(self, key: str, value):
        """
        Set an item in the row

        Args:
            key: The field name
            value: The field value

        """
        self._header.set(self._index, key, value)

    def assign(self, value):
        """
        Assign a row

        Args:
            value: The row of data

        """
        for key in self._header.dtype.fields:
            self[key] = value[key]

    def __array__(self, dtype=METADATA_DTYPE):
        """
        Convert to a numpy array

        """
        result = np.zeros(self.size, dtype=self._header.dtype)
        for key in self._header.dtype.fields:
            result[key] = self[key]
        return result.astype(dtype)


class Column(object):
    """
    An object to represent a column

    """

    def __init__(self, header, key: str):
        self._header = header
        self._key = key

    @property
    def size(self):
        """
        The size of the header

        """
        return 1

    def __getitem__(self, index):
        """
        Get an item from the column

        Args:
            index: The row index

        Returns:
            The element

        """
        return self._header.get(index, self._key)

    def __setitem__(self, index, value):
        """
        Set an item in the column

        Args:
            index: The row index
            value: The field value

        """
        self._header.set(index, self._key, value)

    def __array__(self, dtype=None):
        """
        Convert to a numpy array

        """
        return self[:]


class Header(object):
    """
    An object to represent the header

    """

    def __getitem__(self, item):
        """
        Get a row or column item

        """
        if isinstance(item, str):
            Class = Column
        else:
            Class = Row
        return Class(self, item)

    def __setitem__(self, item, value):
        """
        Set a row or column item

        """
        if isinstance(item, str):
            col = Column(self, item)
            col[:] = value
        else:
            row = Row(self, item)
            row.assign(value)

    def rows(self):
        """
        Get a row iterator

        """
        for i in range(self.size):
            yield Row(self, i)

    def cols(self):
        """
        Get a column iterator

        """
        for key in self.fields:
            yield Column(self, key)

    @property
    def fields(self):
        """
        Get the datatype fields

        """
        return self.dtype.fields

    @property
    def dtype(self):
        """
        Get the metadata datatype

        """
        return METADATA_DTYPE

    @property
    def angle(self) -> np.ndarray:
        """
        An alias to get the angle

        """
        return self["tilt_alpha"]

    @property
    def position(self) -> np.ndarray:
        """
        An alias to get the position

        """
        result = np.zeros(shape=(self.size, 3), dtype=np.float32)
        result[:, 0] = self["stage_x"][:] + self["shift_x"][:]
        result[:, 1] = self["stage_y"][:] + self["shift_y"][:]
        result[:, 2] = self["stage_z"][:]
        return result

    def __array__(self, dtype=METADATA_DTYPE, copy=True):
        """
        Convert to a numpy array

        """
        return np.asarray(self[:])

    def get(self, index, key: str):
        """
        Get the value of a property

        """
        pass

    def set(self, index, key: str, value):
        """
        Set the value of a property

        """
        pass

    @property
    def size(self):
        """
        Get the size of the header

        """
        pass

    @property
    def scan(self):
        """
        Get the scan

        """

        # Get the metadata
        metadata = np.array(self)

        # Construct the orientation
        axis = np.stack(
            [
                metadata["tilt_axis_x"],
                metadata["tilt_axis_y"],
                metadata["tilt_axis_z"],
            ]
        ).T

        # Construct the shift
        shift = np.stack(
            [metadata["shift_x"], metadata["shift_y"], metadata["stage_z"]]
        ).T

        # Construct the shift delta
        shift_delta = np.stack(
            [
                metadata["shift_offset_x"],
                metadata["shift_offset_y"],
                metadata["stage_offset_z"],
            ]
        ).T

        # Return a scan object
        return parakeet.scan.Scan(
            image_number=metadata["image_number"],
            fraction_number=metadata["fraction_number"],
            axis=axis,
            angle=metadata["tilt_alpha"],
            shift=shift,
            shift_delta=shift_delta,
            beam_tilt_theta=metadata["theta"],
            beam_tilt_phi=metadata["phi"],
            exposure_time=metadata["exposure_time"],
        )


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
    """
    A sub class for the MRC file header

    """

    def __init__(self, handle):
        self._handle = handle

    def mapping(self, key: str):
        """
        Get the mapping between names

        """
        assert key in self.dtype.fields
        return {
            #
            # General parameters
            #
            "application": "Application",
            "application_version": "Application version",
            "timestamp": "Timestamp",
            #
            # Scan parameters
            #
            "image_number": "Start frame",
            "fraction_number": "Fraction number",
            #
            # Stage parameters
            #
            "tilt_alpha": "Alpha tilt",
            "tilt_axis_angle": "Tilt axis angle",
            "stage_x": "X-Stage",
            "stage_y": "Y-Stage",
            "stage_z": "Z-Stage",
            #
            # Beam parameters
            #
            "energy": "HT",
            "dose": "Dose",
            "slit_inserted": "Slit inserted",
            "slit_width": "Slit width",
            "energy_shift": "Energy shift",
            "shift_x": "Shift X",
            "shift_y": "Shift Y",
            "shift_offset_x": "Shift offset X",
            "shift_offset_y": "Shift offset Y",
            #
            # Detector parameters
            #
            "pixel_size_x": "Pixel size X",
            "pixel_size_y": "Pixel size Y",
            "gain": "Gain",
            "offset": "Offset",
            #
            # Lens parameters
            #
            "c_10": "Defocus",
            "phase_plate": "Phase Plate",
        }.get(key, None)

    def get(self, index, key: str):
        """
        Get the value of a property

        """
        mapping = self.mapping(key)
        getter = {
            "pixel_size_x": lambda x: x * 1e-10,
            "pixel_size_y": lambda x: x * 1e-10,
        }.get(key, lambda x: x)
        if not mapping:
            return np.zeros(shape=self._handle.size, dtype=self.dtype.fields[key][0])[
                index
            ]
        return getter(self._handle[index][mapping])

    def set(self, index, key: str, value):
        """
        Set the value of a property

        """
        mapping = self.mapping(key)
        setter = {
            "pixel_size_x": lambda x: x * 1e10,
            "pixel_size_y": lambda x: x * 1e10,
        }.get(key, lambda x: x)
        if mapping:
            if isinstance(value, np.ndarray) and len(value) == 1:
                value = value[0]
            self._handle[index][mapping] = setter(value)

    @property
    def size(self) -> int:
        """
        Get the size of the header

        """
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

        # Create the extended header
        extended_header = np.zeros(shape=shape[0], dtype=FEI_EXTENDED_HEADER_DTYPE)
        extended_header["Metadata size"] = extended_header.dtype.itemsize

        # Open the handle to the mrcfile
        self.handle = mrcfile.new_mmap(
            filename,
            shape=shape,
            mrc_mode=mrcfile.utils.mode_from_dtype(dtype),
            overwrite=True,
            extended_header=extended_header,
            exttyp=b"FEI1",
        )

        # Set the data array
        self._data = self.handle.data

        # Create the header info
        self._header = MrcfileHeader(extended_header)

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

    @property
    def particle_positions(self):
        """
        Return the particle positions

        """
        return None

    @particle_positions.setter
    def particle_positions(self, particle_positions):
        """
        Set the particle positions

        """
        pass

    def update(self):
        """
        Update before closing

        """
        self.handle.update_header_stats()


class NexusHeader(Header):
    """
    A sub class for the MRC file header

    """

    def __init__(self, handle):
        self._handle = handle

    def mapping(self, key):
        """
        Get the mapping between names

        """
        assert key in self.dtype.fields
        return key

    def get(self, index, key):
        """
        Get the mapping between names

        """
        mapping = self.mapping(key)
        return self._handle[mapping][index]

    def set(self, index, key, value):
        """
        Set the value of a property

        """
        mapping = self.mapping(key)
        self._handle[mapping][index] = value

    @property
    def size(self):
        """
        Get the size of the header

        """
        return self._handle["data"].shape[0]


class ImageHeader(Header):
    """
    A sub class for an image header

    """

    def __init__(self, handle):
        self._handle = handle

    def mapping(self, key):
        """
        Get the mapping between names

        """
        assert key in self.dtype.fields
        return key

    def get(self, index, key):
        """
        Get the mapping between names

        """
        mapping = self.mapping(key)
        return self._handle[mapping][index]

    def set(self, index, key, value):
        """
        Set the value of a property

        """
        mapping = self.mapping(key)
        self._handle[mapping][index] = value

    @property
    def size(self):
        """
        Get the size of the header

        """
        return self._handle.shape[0]


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

        # Create the sample
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample["name"] = "parakeet-simulation"

        # Create the data
        data = entry.create_group("data")
        data["data"] = detector["data"]

        # Set the data ptr
        self._data = data["data"]
        self._header = NexusHeader(data)

        # Create the datasets for the header
        for key, (dtype, _) in self._header.dtype.fields.items():
            data.create_dataset(key, shape=shape[0], dtype=dtype)

        # Fill a few values
        self._header[:]["pixel_size_x"] = pixel_size
        self._header[:]["pixel_size_y"] = pixel_size
        self._header[:]["application"] = "Parakeet"
        self._header[:]["application_version"] = parakeet.__version__

    @property
    def pixel_size(self):
        """
        Return the pixel size

        """
        return self.handle["data"]["pixel_size_x"][0]

    @property
    def particle_positions(self):
        """
        Return the particle positions

        """
        sample = self.handle["entry"]["sample"]
        if "particle_positions" in sample:
            return sample["particle_positions"][:]
        return None

    @particle_positions.setter
    def particle_positions(self, particle_positions):
        """
        Set the particle positions

        """
        if particle_positions is not None:
            self.handle["entry"]["sample"].create_dataset(
                "particle_positions", data=particle_positions
            )


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
        self._header = ImageHeader(np.zeros(shape=shape[0], dtype=METADATA_DTYPE))

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
        particle_positions,
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
        self.particle_positions = particle_positions
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
            try:
                assert handle.indexed_extended_header.dtype == FEI_EXTENDED_HEADER_DTYPE
                assert len(handle.indexed_extended_header.shape) == 1
                assert handle.indexed_extended_header.shape[0] == handle.data.shape[0]
                extended_header = handle.indexed_extended_header
            except Exception:
                extended_header = np.zeros(
                    shape=handle.data.shape[0], dtype=FEI_EXTENDED_HEADER_DTYPE
                )
        else:
            extended_header = np.zeros(
                shape=handle.data.shape[0], dtype=FEI_EXTENDED_HEADER_DTYPE
            )

        # Set the header
        header = MrcfileHeader(extended_header)

        # Get the pixel size
        pixel_size = float(handle.voxel_size["x"])

        # Get the particle positions
        particle_positions = None

        # Create the reader
        return Reader(
            handle,
            handle.data,
            header,
            pixel_size,
            particle_positions,
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

        # Get the header
        header = NexusHeader(data)

        # Get the pixel size
        pixel_size = data.get("pixel_size_x", [1.0])[0]

        # Get the particle positions
        if "particle_positions" in handle["entry"]["sample"]:
            particle_positions = handle["entry"]["sample"]["particle_positions"]
        else:
            particle_positions = None

        # Create the reader
        return Reader(
            handle,
            data["data"],
            header,
            pixel_size,
            particle_positions,
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
