#
# parakeet.simulate.cbed.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#

import logging
import numpy as np
import time
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.io
import parakeet.sample
import parakeet.simulate
from parakeet.config import Device
from parakeet.simulate.simulation import Simulation
from parakeet.simulate.engine import SimulationEngine
from parakeet.microscope import Microscope
from parakeet.scan import Scan
from functools import singledispatch
from math import pi
from scipy.spatial.transform import Rotation as R


__all__ = ["cbed"]


# Get the logger
logger = logging.getLogger(__name__)


class CBEDImageSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self,
        microscope=None,
        sample=None,
        scan=None,
        simulation=None,
        device="gpu",
        gpu_id=None,
    ):
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

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Get the rotation angle
        image_number = self.scan.image_number[index]
        fraction_number = self.scan.fraction_number[index]
        angle = self.scan.angles[index]
        axis = self.scan.axes[index]
        position = self.scan.position[index]
        orientation = self.scan.orientation[index]
        shift = self.scan.shift[index]
        drift = self.scan.shift_delta[index]
        beam_tilt_theta = self.scan.beam_tilt_theta[index]
        beam_tilt_phi = self.scan.beam_tilt_phi[index]
        exposure_time = self.scan.exposure_time[index]
        electrons_per_angstrom = self.scan.electrons_per_angstrom[index]

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        origin = np.array(self.microscope.detector.origin)
        margin = self.simulation["margin"]
        padding = self.simulation["padding"]
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        # margin_offset = margin * pixel_size
        # padding_offset = padding * pixel_size
        offset = (padding + margin) * pixel_size

        # The Z centre
        z_centre = self.sample.centre[2]

        # Create the multem input multislice object
        simulate = SimulationEngine(
            self.device,
            self.gpu_id,
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"] + self.simulation["padding"],
            "CBED",
            z_centre,
        )

        # Set the specimen size
        simulate.input.spec_lx = x_fov + offset * 2
        simulate.input.spec_ly = y_fov + offset * 2
        simulate.input.spec_lz = self.sample.containing_box[1][2]

        # Set the beam tilt
        simulate.input.theta += beam_tilt_theta
        simulate.input.phi += beam_tilt_phi

        # Compute the B factor
        if self.simulation["radiation_damage_model"]:
            simulate.input.static_B_factor = (
                8
                * pi**2
                * (
                    self.simulation["sensitivity_coefficient"]
                    * electrons_per_angstrom
                    * (index + 1)
                )
            )
        else:
            simulate.input.static_B_factor = 0

        # Set the atoms in the input after translating them for the offset
        atoms = self.sample.get_atoms()
        logger.info("Simulating with %d atoms" % atoms.data.shape[0])
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

        # Select atoms in FOV
        fov_xmin = origin[0] - offset
        fov_xmax = fov_xmin + x_fov + 2 * offset
        fov_ymin = origin[1] - offset
        fov_ymax = fov_ymin + y_fov + 2 * offset
        if len(atoms.data) > 0:
            select = (
                (atoms.data["x"] >= fov_xmin)
                & (atoms.data["x"] <= fov_xmax)
                & (atoms.data["y"] >= fov_ymin)
                & (atoms.data["y"] <= fov_ymax)
            )
            atoms.data = atoms.data[select]

        # Translate for the detector
        simulate.input.spec_atoms = atoms.translate(
            (offset - origin[0], offset - origin[1], 0)
        ).to_multem()
        logger.info("   Got spec atoms")

        if len(atoms.data) > 0:
            print(
                "Atoms X min/max: %.1f, %.1f"
                % (atoms.data["x"].min(), atoms.data["x"].max())
            )
            print(
                "Atoms Y min/max: %.1f, %.1f"
                % (atoms.data["y"].min(), atoms.data["y"].max())
            )
            print(
                "Atoms Z min/max: %.1f, %.1f"
                % (atoms.data["z"].min(), atoms.data["z"].max())
            )

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
            image = simulate.diffraction_image(masker)

        else:
            # Set the incident wave
            simulate.input.iw_x = [0]  # simulate.input.spec_lx/2
            simulate.input.iw_y = [0]  # simulate.input.spec_ly/2

            # Run the simulation
            logger.info("Simulating")
            image = simulate.diffraction_image()

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        x0 = padding
        y0 = padding
        x1 = image.shape[1] - padding
        y1 = image.shape[0] - padding
        image = image[y0:y1, x0:x1]

        # Print some info
        logger.info("Ideal image min/max: %f/%f" % (np.min(image), np.max(image)))

        # Get the timestamp
        timestamp = time.time()

        # Set the metaadata
        metadata = self.metadata[index]
        metadata["image_number"] = image_number
        metadata["fraction_number"] = fraction_number
        metadata["timestamp"] = timestamp
        metadata["tilt_alpha"] = angle
        metadata["tilt_axis_x"] = axis[0]
        metadata["tilt_axis_y"] = axis[1]
        metadata["tilt_axis_z"] = axis[2]
        metadata["shift_x"] = shift[0]
        metadata["shift_y"] = shift[1]
        metadata["stage_z"] = shift[2]
        metadata["shift_offset_x"] = drift[0]
        metadata["shift_offset_y"] = drift[1]
        metadata["stage_offset_z"] = drift[2]
        metadata["energy"] = self.microscope.beam.energy
        metadata["theta"] = self.microscope.beam.theta
        metadata["phi"] = self.microscope.beam.phi
        metadata["image_size_x"] = nx
        metadata["image_size_y"] = ny
        metadata["ice"] = self.simulation["ice"]
        metadata["damage_model"] = self.simulation["radiation_damage_model"]
        metadata["sensitivity_coefficient"] = self.simulation["sensitivity_coefficient"]
        metadata["exposure_time"] = exposure_time
        metadata["dose"] = electrons_per_angstrom

        # Compute the image scaled with Poisson noise
        return (index, image, metadata)


def simulation_factory(
    microscope: Microscope,
    sample: parakeet.sample.Sample,
    scan: Scan,
    simulation: dict = None,
    multiprocessing: dict = None,
) -> Simulation:
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        sample (object): The sample object
        scan (object): The scan object
        simulation (object): The simulation parameters
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
        # assert multiprocessing["nproc"] in [None, 1]
        assert len(multiprocessing["gpu_id"]) == 1

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * margin,
            microscope.detector.ny + 2 * margin,
        ),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        simulate_image=CBEDImageSimulator(
            microscope=microscope,
            sample=sample,
            scan=scan,
            simulation=simulation,
            device=multiprocessing["device"],
            gpu_id=multiprocessing["gpu_id"][0],
        ),
    )


@singledispatch
def cbed(
    config_file,
    sample_file: str,
    image_file: str,
    device: Device = None,
    nproc: int = None,
    gpu_id: list = None,
):
    """
    Simulate the image from the sample

    Args:
        config_file: The config filename
        sample_file: The sample filename
        image_file: The image filename
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

    # Create the sample
    logger.info(f"Loading sample from {sample_file}")
    sample = parakeet.sample.load(sample_file)

    # The image file
    _cbed_Config(config, sample, image_file)


@cbed.register(parakeet.config.Config)
def _cbed_Config(
    config: parakeet.config.Config, sample: parakeet.sample.Sample, image_file: str
):
    """
    Simulate the image from the sample

    Args:
        config: The config object
        sample: The sample object
        image_file: The image filename

    """

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

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
        microscope,
        sample,
        scan,
        simulation=config.simulation.model_dump(),
        multiprocessing=config.multiprocessing.model_dump(),
    )

    # Create the writer
    logger.info(f"Opening file: {image_file}")
    writer = parakeet.io.new(
        image_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.float32,
    )

    # Run the simulation
    simulation.simulate_image.metadata = writer.header
    simulation.run(writer)
