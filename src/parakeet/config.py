#
# parakeet.config.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import copy
import logging
import yaml

from enum import Enum
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# Get the logger
logger = logging.getLogger(__name__)


def temp_directory():
    """
    Returns:
        str: A temp directory for working in

    """
    return "_parakeet"


class BaseModel(PydanticBaseModel):
    """
    Create a custom base to define desired behaviour

    """

    class Config:

        # Ensure that enums use string values
        use_enum_values = True

        # Don't allow extra fields
        extra = "forbid"


class Auto(str, Enum):
    """
    An enumeration just containing auto

    """

    auto = "auto"


class ShapeType(str, Enum):
    """
    An enumeration of sample shape types

    """

    cube = "cube"
    cuboid = "cuboid"
    cylinder = "cylinder"


class Cube(BaseModel):
    """
    A model of a cubic sample shape

    """

    length: float = Field(0, description="The cube side length (A)", gt=0)


class Cuboid(BaseModel):
    """
    A model of a cuboid sample shape

    """

    length_x: float = Field(0, description="The cuboid X side length (A)", gt=0)
    length_y: float = Field(0, description="The cuboid Y side length (A)", gt=0)
    length_z: float = Field(0, description="The cuboid Z side length (A)", gt=0)


class Cylinder(BaseModel):
    """
    A model of a cylindrical sample shape

    """

    length: float = Field(0, description="The cylinder length (A)", gt=0)

    radius: Union[float, List[float]] = Field(
        0, description="The cylinder radius (A)", gt=0
    )

    axis: Tuple[float, float, float] = Field(
        (0, 1, 0), description="The axis of the cylinder"
    )

    offset_x: List[float] = Field(
        None, description="The x offset as a function of cylinder y position"
    )

    offset_z: List[float] = Field(
        None, description="The z offset as a function of cylinder y position"
    )


class Shape(BaseModel):
    """
    A model to describe the sample shape

    """

    type: ShapeType = Field("cube", description="The shape of the sample")

    cube: Cube = Field(
        Cube(length=1000),
        description="The parameters of the cubic sample (only used if type == cube)",
    )

    cuboid: Cuboid = Field(
        Cuboid(length_x=1000, length_y=1000, length_z=1000),
        description=(
            "The parameters of the cuboid sample (only used if type == " "cuboid)"
        ),
    )

    cylinder: Cylinder = Field(
        Cylinder(length=1000, radius=500),
        description=(
            "The parameters of the cylindrical sample (only used if type == "
            "cylinder)"
        ),
    )

    margin: Tuple[float, float, float] = Field(
        (0, 0, 0),
        description=(
            "The shape margin used to define how close to the edges particles "
            "should be placed (A)"
        ),
    )


class MoleculePose(BaseModel):
    """
    A model to describe a molecule position and orientation

    """

    position: Optional[Tuple[float, float, float]] = Field(
        description=(
            "The molecule position (A, A, A). Setting this to null or an "
            "empty list will cause parakeet to give a random position"
        ),
        examples=[
            "position: null # Assign random position",
            "position: [] # Assign random position",
            "position: [1, 2, 3] # Assign known position",
        ],
    )

    orientation: Optional[Tuple[float, float, float]] = Field(
        description=(
            "The molecule orientation defined as a rotation vector. Setting "
            "this to null or an empty list will cause parakeet to give a "
            "random orientation"
        ),
        examples=[
            "orienation: null # Assign random orienation",
            "orienation: [] # Assign random orienation",
            "orienation: [1, 2, 3] # Assign known orienation",
        ],
    )


class CoordinateFile(BaseModel):
    """
    A model to describe a local coordinate file

    """

    filename: str = Field(
        None, description="The filename of the atomic coordinates to use (*.pdb, *.cif)"
    )

    recentre: bool = Field(True, description="Recentre the coordinates")


class LocalMolecule(BaseModel):
    """
    A model to describe a local molecule and its instances

    """

    filename: str = Field(
        description="The filename of the atomic coordinates to use (*.pdb, *.cif)"
    )

    instances: Union[int, List[MoleculePose]] = Field(
        1,
        description=(
            "The instances of the molecule to put into the sample model. This "
            "field can be set as either an integer or a list of MoleculePose "
            "objects. If it is set to an integer == 1 then the molecule will be "
            "positioned in the centre of the sample volume; any other integer "
            "will result in the molecules being positioned at random positions "
            "and orientations in the volume. If a list of MoleculePose objects "
            "is given then an arbitrary selection of random and assigned "
            "positions and poses can be set"
        ),
        examples=[
            "instances: 1 # Position 1 molecule at the centre of the sample volume",
            "instances: 10 # Position 10 molecules at random",
            "instances: [ { position: [1, 2, 3], orientation: [4, 5, 6] } ]",
        ],
    )


class PDBMolecule(BaseModel):
    """
    A model to describe a PDB molecule and its instances

    """

    id: str = Field(
        description="The PDB ID of the atomic coordinates to use (*.pdb, *.cif)"
    )

    instances: Union[int, List[MoleculePose]] = Field(
        1,
        description=(
            "The instances of the molecule to put into the sample model. This "
            "field can be set as either an integer or a list of MoleculePose "
            "objects. If it is set to an integer == 1 then the molecule will be "
            "positioned in the centre of the sample volume; any other integer "
            "will result in the molecules being positioned at random positions "
            "and orientations in the volume. If a list of MoleculePose objects "
            "is given then an arbitrary selection of random and assigned "
            "positions and poses can be set"
        ),
        examples=[
            "instances: 1 # Position 1 molecule at the centre of the sample volume",
            "instances: 10 # Position 10 molecules at random",
            "instances: [ { position: [1, 2, 3], orientation: [4, 5, 6] } ]",
        ],
    )


class Molecules(BaseModel):
    """
    A model to describe the molecules to add to the sample

    """

    local: Optional[List[LocalMolecule]] = Field(
        description="The local molecules to include in the sample model"
    )

    pdb: Optional[List[PDBMolecule]] = Field(
        description="The PDB molecules to include in the sample model"
    )


class Ice(BaseModel):
    """
    A model to describe a uniform random atomic ice model. If generate is True
    then generate random water positions with a given density. It is usually
    better to use the Gaussian Random Field (GRF) ice model which can be set in
    the simulation model.

    """

    generate: bool = Field(
        False, description="Generate the atomic ice model (True/False)"
    )

    density: float = Field(940, description="The density of the ice (Kg/m3)")


class Sputter(BaseModel):
    """
    A model to describe a sputter coating to the sample

    """

    element: str = Field(
        description="The symbol of the atom for the sputter coating material"
    )

    thickness: float = Field(description="The thickness of the sputter coating (A)")


class Sample(BaseModel):
    """
    A model to describe the sample

    """

    shape: Shape = Field(Shape(), description="The shape parameters of the sample")

    box: Tuple[float, float, float] = Field(
        (1000, 1000, 1000), description="The sample box (A, A, A)"
    )

    centre: Tuple[float, float, float] = Field(
        (500, 500, 500), description="The centre of rotation (A, A, A)"
    )

    coords: Optional[CoordinateFile] = Field(
        description="Coordinates to initialise the sample"
    )

    molecules: Optional[Molecules] = Field(
        description="The molecules to include in the sample model"
    )

    ice: Optional[Ice] = Field(description="The atomic ice model parameters.")

    sputter: Optional[Sputter] = Field(
        description="The sputter coating model parameters."
    )


class Beam(BaseModel):
    """
    A model to describe the beam

    """

    energy: float = Field(300, description="The electron energy (keV)")

    energy_spread: float = Field(2.66e-6, description="The energy spread (dE/E)")

    acceleration_voltage_spread: float = Field(
        0.8e-6, description="The acceleration voltage spread (dV/V)"
    )

    electrons_per_angstrom: float = Field(
        30, description="The number of electrons per square angstrom"
    )

    illumination_semiangle: float = Field(
        0.02, description="The illumination semiangle (mrad)."
    )

    theta: float = Field(0, description="The beam tilt theta angle (deg)")

    phi: float = Field(0, description="The beam tilt phi angle (deg)")


class Lens(BaseModel):
    """
    A model to describe the objective lens

    """

    c_10: float = Field(-20000, description="The defocus (A). Negative is underfocus.")
    c_12: float = Field(0, description="The 2-fold astigmatism (A)")
    phi_12: float = Field(
        0, description="The Azimuthal angle of 2-fold astigmatism (rad)"
    )

    c_21: float = Field(0, description="The Axial coma (A)")
    phi_21: float = Field(0, description="The Azimuthal angle of axial coma (rad)")
    c_23: float = Field(0, description="The 3-fold astigmatism (A)")
    phi_23: float = Field(
        0, description="The Azimuthal angle of 3-fold astigmatism (rad)"
    )

    c_30: float = Field(2.7, description="The 3rd order spherical aberration (mm)")
    c_32: float = Field(0, description="The Axial star aberration (A)")
    phi_32: float = Field(
        0, description="The Azimuthal angle of axial star aberration (rad)"
    )
    c_34: float = Field(0, description="The 4-fold astigmatism (A)")
    phi_34: float = Field(
        0, description="The Azimuthal angle of 4-fold astigmatism (rad)"
    )

    c_41: float = Field(0, description="The 4th order axial coma (A)")
    phi_41: float = Field(
        0, description="The Azimuthal angle of 4th order axial coma (rad)"
    )
    c_43: float = Field(0, description="The 3-lobe aberration (A)")
    phi_43: float = Field(
        0, description="The Azimuthal angle of 3-lobe aberration (rad)"
    )
    c_45: float = Field(0, description="The 5-fold astigmatism (A)")
    phi_45: float = Field(
        0, description="The Azimuthal angle of 5-fold astigmatism (rad)"
    )

    c_50: float = Field(0, description="The 5th order spherical aberration (A)")
    c_52: float = Field(0, description="The 5th order axial star aberration (A)")
    phi_52: float = Field(
        0, description="The Azimuthal angle of 5th order axial star aberration (rad)"
    )
    c_54: float = Field(0, description="The 5th order rosette aberration (A)")
    phi_54: float = Field(
        0, description="The Azimuthal angle of 5th order rosette aberration (rad)"
    )
    c_56: float = Field(0, description="The 6-fold astigmatism (A)")
    phi_56: float = Field(
        0, description="The Azimuthal angle of 6-fold astigmatism (rad)"
    )

    c_c: float = Field(2.7, description="The chromatic aberration (mm)")

    current_spread: float = Field(0.33e-6, description="The current spread (dI/I)")


class Detector(BaseModel):
    """
    A model to describe the detector

    """

    nx: int = Field(1000, description="The number of pixels in X")

    ny: int = Field(1000, description="The number of pixels in Y")

    pixel_size: float = Field(1, description="The pixel size (A)")

    dqe: bool = Field(False, description="Use the DQE model (True/False)")

    origin: Tuple[int, int] = Field(
        (0, 0), description="The origin of the detector in lab space(A,A)"
    )


class MicroscopeModel(str, Enum):
    """
    An enumeration to describe the microscope model

    """

    krios = "krios"
    talos = "talos"


class Microscope(BaseModel):
    """
    A model to describe the microscope

    """

    model: MicroscopeModel = Field(
        None, description="Use parameters for a given microscope model"
    )

    beam: Beam = Field(Beam(), description="The beam model parameters")

    lens: Lens = Field(Lens(), description="The lens model parameters")

    phase_plate: bool = Field(False, description="Use a phase plate (True/False)")

    detector: Detector = Field(Detector(), description="The detector model parameters")


class ScanMode(str, Enum):
    """
    An enumeration to describe the scan mode

    """

    manual = "manual"
    still = "still"
    tilt_series = "tilt_series"
    dose_symmetric = "dose_symmetric"
    single_particle = "single_particle"
    helical_scan = "helical_scan"
    nhelix = "nhelix"
    beam_tilt = "beam_tilt"


class Drift(BaseModel):
    """
    A model to describe the beam drift

    """

    magnitude: float = Field(0, description="The magnitude of the drift (A)")

    kernel_size: int = Field(0, description="How much to smooth the drift")


class Scan(BaseModel):
    """
    A model to describe the scan

    """

    mode: ScanMode = Field("still", description="Set the scan mode")

    axis: Tuple[float, float, float] = Field(
        (0, 1, 0), description="The scan axis vector"
    )

    start_angle: float = Field(0, description="The start angle for the rotation (deg)")

    step_angle: float = Field(0, description="The step angle for the rotation (deg)")

    start_pos: float = Field(
        0, description="The start position for a translational scan (A)"
    )

    step_pos: Union[float, Auto] = Field(
        "auto", description="The step distance for a translational scan (A)"
    )

    num_images: int = Field(1, description="The number of images to simulate")

    num_nhelix: int = Field(1, description="The number of scans in an n-helix")

    exposure_time: float = Field(1, description="The exposure time per image (s)")

    angles: Optional[List[float]] = Field(
        None,
        description=(
            "The list of angles to use (deg). This field is used when the mode"
            "is set to 'manual' or 'beam tilt'."
        ),
    )

    positions: Optional[List[float]] = Field(
        None,
        description=(
            "The list of positions to use (A). This field is used when the mode"
            "is set to 'manual' or 'beam tilt'."
        ),
    )

    theta: Optional[Union[float, List[float]]] = Field(
        None,
        description=(
            "The list of theta angles to use (mrad) for the beam tilt."
            "This must either be the same length as phi or a scalar"
        ),
    )

    phi: Optional[Union[float, List[float]]] = Field(
        None,
        description=(
            "The list of phi angles to use (mrad) for the beam tilt."
            "This must either be the same length as theta or a scalar"
        ),
    )

    drift: Optional[Drift] = Field(description="The drift model parameters")


class InelasticModel(str, Enum):
    """
    A model to describe the inelastic scattering mode

    """

    zero_loss = "zero_loss"
    mp_loss = "mp_loss"
    unfiltered = "unfiltered"
    cc_corrected = "cc_corrected"


class MPLPosition(str, Enum):
    """
    A model to describe the MPL position mode

    """

    peak = "peak"
    optimal = "optimal"


class Simulation(BaseModel):
    """
    A model to describe the simulation parameters

    """

    slice_thickness: float = Field(3.0, description="The multislice thickness (A)")

    margin: int = Field(100, description="The margin around the image")

    padding: int = Field(100, description="Additional padding")

    division_thickness: int = Field(100, description="Deprecated")

    ice: bool = Field(
        False, description="Use the Gaussian Random Field ice model (True/False)"
    )

    radiation_damage_model: bool = Field(
        False, description="Use the radiation damage model (True/False)"
    )

    inelastic_model: InelasticModel = Field(
        None, description="The inelastic model parameters"
    )

    mp_loss_width: float = Field(None, description="The MPL energy filter width")

    mp_loss_position: MPLPosition = Field(
        "peak", description="The MPL energy filter position"
    )

    sensitivity_coefficient: float = Field(
        0.022, description="The radiation damage model sensitivity coefficient"
    )


class ClusterMethod(str, Enum):
    """
    An enumeration to describe the cluster method

    """

    sge = "sge"


class Cluster(BaseModel):
    """
    A model to set the cluster parameters for multiprocessing

    """

    method: ClusterMethod = Field(None, description="The cluster method to use")

    max_workers: int = Field(1, description="The maximum number of worker processes")


class Device(str, Enum):
    """
    An enumeration to set whether to run on the GPU or CPU

    """

    gpu = "gpu"
    cpu = "cpu"


class Config(BaseModel):
    """
    The Parakeet configuration parameters

    """

    sample: Sample = Field(
        Sample(),
        description="The sample parameters",
    )

    microscope: Microscope = Field(
        Microscope(),
        description="The microscope parameters",
    )

    scan: Scan = Field(Scan(), description="The scan parameters")

    simulation: Simulation = Field(
        Simulation(), description="The simulation parameters"
    )

    device: Device = Field("gpu", description="The device to use (cpu or gpu)")

    cluster: Cluster = Field(Cluster(), description="The cluster parameters")


def default() -> Config:
    """
    Return:
        obj: the default configuration

    """
    return Config()


def save(config: Config, filename: str = "config.yaml", **kwargs):
    """
    Save the configuration file

    Args:
        config (str): The configuration object
        filename (str): The configuration filename

    """

    # Get the dictionary
    d = config.dict(**kwargs)

    # Write the output file
    with open(filename, "w") as outfile:
        yaml.safe_dump(d, outfile)


def load(config: Union[str, dict] = None) -> Config:
    """
    Load the configuration from the various inputs

    Args:
        config (str): The config filename or config dictionary

    Returns:
        dict: The configuration dictionary

    """

    # If the yaml configuration is set then merge the configuration
    if config:
        if isinstance(config, str):
            with open(config) as infile:
                config_file = yaml.safe_load(infile)
        else:
            config_file = config
    else:
        config_file = {}

    # Get the configuration
    return Config(**config_file)


def new(filename: str = "config.yaml", full: bool = False) -> Config:
    """
    Generate a new config file

    Args:
        filename: The config filename
        full: Full or basic configuration

    """

    # Get the configuration object
    config = Config()

    # Set items to include in output
    if full:
        include = None
    else:
        include = {
            "microscope": {
                "beam": {"electrons_per_angstrom", "energy", "illumination_semiangle"},
                "detector": {
                    "nx",
                    "ny",
                    "pixel_size",
                },
                "lens": {
                    "c_10",
                    "c_30",
                    "c_c",
                },
            },
            "sample": {
                "box",
                "centre",
                "molecules",
                "shape",
            },
            "scan": {
                "mode",
                "num_images",
                "start_angle",
                "step_angle",
            },
            "simulation": {"ice"},
        }

    # Save the config file
    save(config, filename, include=include)

    # Return the config
    return config


def edit(
    in_filename: str = "config.yaml", out_filename: str = None, config_obj: str = ""
):
    """
    Edit the configuration

    """

    def get_config_obj(config_obj):
        if isinstance(config_obj, str):
            return yaml.safe_load(config_obj)
        return config_obj

    # Check the output filename
    if out_filename is None:
        out_filename = in_filename

    # Parse the arguments
    config = load(in_filename)

    # Merge the dictionaries
    d1 = config.dict(exclude_unset=True)
    d2 = get_config_obj(config_obj)
    d = deepmerge(d1, d2)

    # # Load the new configuration
    config = load(d)

    # Save the config
    save(config, out_filename, exclude_unset=True)

    # Return config
    return config


def show(config: Config, full: bool = False):
    """
    Print the command line arguments

    Args:
        config: The configuration object
        full: Show the full configuration (True or False)

    """
    return yaml.safe_dump(config.dict(exclude_unset=not full), indent=4)


def deepmerge(a: dict, b: dict) -> dict:
    """
    Perform a deep merge of two dictionaries

    Args:
        a: The first dictionary
        b: The second dictionary
    Returns:
        The merged dictionary

    """

    def deepmerge_internal(self, other):
        for key, value in other.items():
            if key in self:
                if isinstance(value, dict):
                    if self[key] is None:
                        self[key] = {}
                    deepmerge_internal(self[key], value)
                else:
                    self[key] = copy.deepcopy(value)
            else:
                self[key] = copy.deepcopy(value)
        return self

    return deepmerge_internal(copy.deepcopy(a), b)
