#
# parakeet.simulate.potential.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#

import logging
import mrcfile
import numpy as np
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.sample
from parakeet.config import Device
from parakeet.microscope import Microscope
from parakeet.scan import Scan
from parakeet.simulate.simulation import Simulation
from parakeet.simulate.engine import SimulationEngine
from functools import singledispatch
from math import pi, floor
from scipy.spatial.transform import Rotation as R


__all__ = ["potential"]


# Get the logger
logger = logging.getLogger(__name__)


class ProjectedPotentialSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self,
        potential_prefix="potential_",
        microscope=None,
        sample=None,
        scan=None,
        simulation=None,
        device="gpu",
        gpu_id=None,
    ):
        self.potential_prefix = potential_prefix
        self.microscope = microscope
        self.sample = sample
        self.scan = scan
        self.simulation = simulation
        self.device = device
        self.gpu_id = gpu_id

    def __call__(self, index):
        """
        Simulate a single frame

        Args:
            simulation (object): The simulation object
            index (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        # Get the rotation angle
        angle = self.scan.angles[index]
        position = self.scan.position[index]
        orientation = self.scan.orientation[index]
        shift = self.scan.shift[index]
        drift = self.scan.shift_delta[index]
        beam_tilt_theta = self.scan.beam_tilt_theta[index]
        beam_tilt_phi = self.scan.beam_tilt_phi[index]

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        margin = self.simulation["margin"]
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        offset = margin * pixel_size

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Create the sample extractor
        x0 = (-offset, -offset)
        x1 = (x_fov + offset, y_fov + offset)

        # The Z centre
        z_centre = self.sample.centre[2]

        # Create the multem input multislice object
        simulate = SimulationEngine(
            self.device,
            self.gpu_id,
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "EWRS",
            z_centre,
        )

        # Set the specimen size
        simulate.input.spec_lx = x_fov + offset * 2
        simulate.input.spec_ly = y_fov + offset * 2
        simulate.input.spec_lz = self.sample.containing_box[1][2]

        # Set the atoms in the input after translating them for the offset
        atoms = self.sample.get_atoms_in_fov(x0, x1)
        logger.info("Simulating with %d atoms" % atoms.data.shape[0])

        # Set atom sigma
        # atoms.data["sigma"] = sigma_B

        if len(atoms.data) > 0:
            coords = atoms.data[["x", "y", "z"]].to_numpy()
            coords = (
                R.from_rotvec(orientation).apply(coords - self.sample.centre)
                + self.sample.centre
                - position
            ).astype("float32")
            atoms.data["x"] = coords[:, 0]
            atoms.data["y"] = coords[:, 1]
            atoms.data["z"] = coords[:, 2]

        origin = (0, 0)
        simulate.input.spec_atoms = atoms.translate(
            (offset - origin[0], offset - origin[1], 0)
        ).to_multem()
        logger.info("   Got spec atoms")

        # Get the potential and thickness
        volume_z0 = self.sample.shape_box[0][2]
        volume_z1 = self.sample.shape_box[1][2]
        slice_thickness = self.simulation["slice_thickness"]
        zsize = int(floor((volume_z1 - volume_z0) / slice_thickness) + 1)
        potential = mrcfile.new_mmap(
            "%s_%d.mrc" % (self.potential_prefix, index),
            shape=(zsize, ny, nx),
            mrc_mode=mrcfile.utils.mode_from_dtype(np.dtype(np.float32)),
            overwrite=True,
        )
        potential.voxel_size = tuple((pixel_size, pixel_size, slice_thickness))

        # Simulate the potential
        if self.simulation["ice"] == True:
            # Get the masker
            masker = simulate.masker(
                index,
                pixel_size,
                origin,
                offset,
                orientation,
                position,
                self.sample,
                self.scan,
                self.simulation,
            )

            # Run the simulation
            image = simulate.potential(potential, volume_z0, masker)

        else:
            # Run the simulation
            logger.info("Simulating")
            simulate.potential(potential, volume_z0)

        # Compute the image scaled with Poisson noise
        return (index, None, None)


def simulation_factory(
    potential_prefix: str,
    microscope: Microscope,
    sample: parakeet.sample.Sample,
    scan: Scan,
    simulation: dict = None,
    multiprocessing: dict = None,
) -> Simulation:
    """
    Create the simulation

    Args:
        potential_prefix: The filename prefix
        microscope: The microscope object
        sample: The sample object
        scan: The scan object
        simulation: The simulation parameters
        multiprocessing (object): The multiprocessing parameters

    Returns:
        object: The simulation object

    """
    # Get the margin
    margin = 0 if simulation is None else simulation.get("margin", 0)

    # Check multiprocessing settings
    if multiprocessing is None:
        multiprocessing = {"device": "gpu", "nproc": 1, "gpu_id": 0}
    else:
        assert multiprocessing["nproc"] in [None, 1]
        assert len(multiprocessing["gpu_id"]) == 1

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * margin,
            microscope.detector.ny + 2 * margin,
        ),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        simulate_image=ProjectedPotentialSimulator(
            potential_prefix=potential_prefix,
            microscope=microscope,
            sample=sample,
            scan=scan,
            simulation=simulation,
            device=multiprocessing["device"],
            gpu_id=multiprocessing["gpu_id"][0],
        ),
    )


@singledispatch
def potential(
    config_file,
    sample_file: str,
    potential_prefix: str,
    device: Device = None,
    nproc: int = None,
    gpu_id: list = None,
):
    """
    Simulate the projected potential from the sample

    Args:
        config_file: The input config filename
        sample_file: The input sample filename
        potential_prefix: The input potential filename
        device: The device to run on (cpu or gpu)
        nproc: The number of processes
        gpu_id: The list of gpu ids

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the command line args in a dict
    if device is not None:
        config.multiprocessing.device = device
    if nproc is not None:
        config.multiprocessing.nproc = nproc
    if gpu_id is not None:
        config.multiprocessing.gpu_id = gpu_id

    # Print some options
    parakeet.config.show(config)

    # Do the work
    _potential_Config(config, sample_file, potential_prefix)


@potential.register(parakeet.config.Config)
def _potential_Config(
    config: parakeet.config.Config, sample_file: str, potential_prefix: str
):
    """
    Simulate the projected potential from the sample

    Args:
        config: The input config
        sample_file: The input sample filename
        potential_prefix: The input potential filename

    """

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Create the sample
    logger.info(f"Loading sample from {sample_file}")
    sample = parakeet.sample.load(sample_file)

    # Create the scan
    if config.scan.step_pos == "auto":
        radius = sample.shape_radius
        config.scan.step_pos = config.scan.step_angle * radius * pi / 180.0
    scan = parakeet.scan.new(
        electrons_per_angstrom=microscope.beam.electrons_per_angstrom,
        **config.scan.model_dump(),
    )

    # Create the simulation
    simulation = simulation_factory(
        potential_prefix=potential_prefix,
        microscope=microscope,
        sample=sample,
        scan=scan,
        simulation=config.simulation.model_dump(),
        multiprocessing=config.multiprocessing.model_dump(),
    )

    # Run the simulation
    simulation.run()
