#
# parakeet.simulate.exit_wave.py
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
import warnings
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.io
import parakeet.sample
import parakeet.simulate
from parakeet.simulate.simulation import Simulation
from parakeet.microscope import Microscope
from parakeet.scan import Scan
from functools import singledispatch
from math import pi, sin
from collections.abc import Iterable


__all__ = ["exit_wave"]


Device = parakeet.config.Device
ClusterMethod = parakeet.config.ClusterMethod
Sample = parakeet.sample.Sample

# Get the logger
logger = logging.getLogger(__name__)

# Try to input MULTEM
try:
    import multem
except ImportError:
    warnings.warn("Could not import MULTEM")


class ExitWaveImageSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self, microscope=None, sample=None, scan=None, simulation=None, device="gpu"
    ):
        self.microscope = microscope
        self.sample = sample
        self.scan = scan
        self.simulation = simulation
        self.device = device

    def get_beam_drift(self, index, angle):
        """
        Get the beam drift

        Returns:
            tuple: shiftx, shifty - the beam drift
        """

        beam_drift = self.microscope.beam.drift
        driftx = 0
        drifty = 0
        if beam_drift and not beam_drift["type"] is None:
            if beam_drift["type"] == "random":
                driftx, drifty = np.random.normal(0, beam_drift["magnitude"], size=2)
                logger.info("Adding drift of %f, %f " % (driftx, drifty))
            elif beam_drift["type"] == "random_smoothed":
                if index == 0:

                    def generate_smoothed_random(magnitude, num_images):
                        drift = np.random.normal(0, magnitude, size=(num_images, 2))
                        driftx = np.convolve(drift[:, 0], np.ones(5) / 5, mode="same")
                        drifty = np.convolve(drift[:, 1], np.ones(5) / 5, mode="same")
                        return driftx, drifty

                    self._beam_drift = generate_smoothed_random(
                        beam_drift["magnitude"], len(self.scan)
                    )
                driftx = self._beam_drift[0][index]
                drifty = self._beam_drift[1][index]
                logger.info("Adding drift of %f, %f " % (driftx, drifty))
            elif beam_drift["type"] == "sinusoidal":
                driftx = sin(angle * pi / 180) * beam_drift["magnitude"]
                drifty = driftx
                logger.info("Adding drift of %f, %f " % (driftx, drifty))
            else:
                raise RuntimeError("Unknown drift type")

        # Return the beam drift
        return driftx, drifty

    def get_masker(self, index, input_multislice, pixel_size, drift, origin, offset):
        """
        Get the masker object for the ice specification

        """

        # Create the masker
        masker = multem.Masker(input_multislice.nx, input_multislice.ny, pixel_size)

        # Get the sample centre
        shape = self.sample.shape
        centre = np.array(self.sample.centre)
        drift = np.array(drift)
        shift = self.scan.poses.shifts[index]
        detector_origin = np.array([origin[0], origin[1], 0])
        centre = centre + offset - drift - detector_origin - shift

        # Set the shape
        if shape["type"] == "cube":
            length = shape["cube"]["length"]
            masker.set_cuboid(
                (
                    centre[0] - length / 2,
                    centre[1] - length / 2,
                    centre[2] - length / 2,
                ),
                (length, length, length),
            )
        elif shape["type"] == "cuboid":
            length_x = shape["cuboid"]["length_x"]
            length_y = shape["cuboid"]["length_y"]
            length_z = shape["cuboid"]["length_z"]
            masker.set_cuboid(
                (
                    centre[0] - length_x / 2,
                    centre[1] - length_y / 2,
                    centre[2] - length_z / 2,
                ),
                (length_x, length_y, length_z),
            )
        elif shape["type"] == "cylinder":
            radius = shape["cylinder"]["radius"]
            if not isinstance(radius, Iterable):
                radius = [radius]
            length = shape["cylinder"]["length"]
            offset_x = shape["cylinder"].get("offset_x", [0] * len(radius))
            offset_z = shape["cylinder"].get("offset_z", [0] * len(radius))
            axis = shape["cylinder"].get("axis", (0, 1, 0))
            masker.set_cylinder(
                (centre[0], centre[1] - length / 2, centre[2]),
                axis,
                length,
                list(radius),
                list(offset_x),
                list(offset_z),
            )

        # Rotate unless we have a single particle type simulation
        if self.scan.is_uniform_angular_scan:
            masker.set_rotation(centre, (0, 0, 0))
        else:
            masker.set_rotation(centre, self.scan.poses.orientations[index].as_rotvec())

        # Get the masker
        return masker

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
        angle = self.scan.angles[index]
        position = self.scan.positions[index]

        # Add the beam drift
        driftx, drifty = self.get_beam_drift(index, angle)

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

        # Create the multem system configuration
        system_conf = parakeet.simulate.simulation.create_system_configuration(
            self.device
        )

        # The Z centre
        z_centre = self.sample.centre[2]

        # Create the multem input multislice object
        input_multislice = parakeet.simulate.simulation.create_input_multislice(
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"] + self.simulation["padding"],
            "EWRS",
            z_centre,
        )

        # Set the specimen size
        input_multislice.spec_lx = x_fov + offset * 2
        input_multislice.spec_ly = y_fov + offset * 2
        input_multislice.spec_lz = self.sample.containing_box[1][2]

        # Compute the B factor
        if self.simulation["radiation_damage_model"]:
            input_multislice.static_B_factor = (
                8
                * pi**2
                * (
                    self.simulation["sensitivity_coefficient"]
                    * self.microscope.beam.electrons_per_angstrom
                    * (index + 1)
                )
            )
        else:
            input_multislice.static_B_factor = 0

        # Set the atoms in the input after translating them for the offset
        atoms = self.sample.get_atoms()
        logger.info("Simulating with %d atoms" % atoms.data.shape[0])
        if len(atoms.data) > 0:
            coords = atoms.data[["x", "y", "z"]].to_numpy()
            coords = (
                self.scan.poses.orientations[index].apply(coords - self.sample.centre)
                + self.sample.centre
                - self.scan.poses.shifts[index]
                + np.array([driftx, drifty, 0])
            ).astype("float32")
            atoms.data["x"] = coords[:, 0]
            atoms.data["y"] = coords[:, 1]
            atoms.data["z"] = coords[:, 2]

        input_multislice.spec_atoms = atoms.translate(
            (offset - origin[0], offset - origin[1], 0)
        ).to_multem()
        logger.info("   Got spec atoms")

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
            masker = self.get_masker(
                index, input_multislice, pixel_size, (driftx, drifty, 0), origin, offset
            )

            # Run the simulation
            output_multislice = multem.simulate(system_conf, input_multislice, masker)

        else:

            # Run the simulation
            logger.info("Simulating")
            output_multislice = multem.simulate(system_conf, input_multislice)

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        image = np.array(output_multislice.data[0].psi_coh).T
        image = image[padding:-padding, padding:-padding]

        # Print some info
        psi_tot = np.abs(image) ** 2
        logger.info("Ideal image min/max: %f/%f" % (np.min(psi_tot), np.max(psi_tot)))

        # Get the timestamp
        timestamp = time.time_ns() / 1e9

        # Set the metaadata
        metadata = self.metadata[index]
        metadata["timestamp"] = timestamp
        metadata["tilt_alpha"] = angle
        metadata["stage_z"] = position[2]
        metadata["shift_x"] = position[0]
        metadata["shift_y"] = position[1]
        metadata["shift_offset_x"] = driftx
        metadata["shift_offset_y"] = drifty
        metadata["energy"] = self.microscope.beam.energy
        metadata["theta"] = self.microscope.beam.theta
        metadata["phi"] = self.microscope.beam.phi
        metadata["image_size_x"] = nx
        metadata["image_size_y"] = ny
        metadata["ice"] = self.simulation["ice"]
        metadata["damage_model"] = self.simulation["radiation_damage_model"]
        metadata["sensitivity_coefficient"] = self.simulation["sensitivity_coefficient"]

        # Compute the image scaled with Poisson noise
        return (index, image, metadata)


def simulation_factory(
    microscope: Microscope,
    sample: Sample,
    scan: Scan,
    device: Device = Device.gpu,
    simulation: dict = None,
    cluster: dict = None,
) -> Simulation:
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        sample (object): The sample object
        scan (object): The scan object
        device (str): The device to use
        simulation (object): The simulation parameters
        cluster (object): The cluster parameters

    Returns:
        object: The simulation object

    """
    # Get the margin
    margin = 0 if simulation is None else simulation.get("margin", 0)

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * margin,
            microscope.detector.ny + 2 * margin,
        ),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        cluster=cluster,
        simulate_image=ExitWaveImageSimulator(
            microscope=microscope,
            sample=sample,
            scan=scan,
            simulation=simulation,
            device=device,
        ),
    )


@singledispatch
def exit_wave(
    config_file,
    sample_file: str,
    exit_wave_file: str,
    device: Device = Device.gpu,
    cluster_method: ClusterMethod = None,
    cluster_max_workers: int = 1,
):
    """
    Simulate the exit wave from the sample

    Args:
        config_file: The config filename
        sample_file: The sample filename
        exit_wave_file: The exit wave filename
        device: The device to run on (CPU or GPU)
        cluster_method: The cluster method to use (default None)
        cluster_max_workers: The maximum number of cluster jobs

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the command line args in a dict
    if device is not None:
        config.device = device
    if cluster_max_workers is not None:
        config.cluster.max_workers = cluster_max_workers
    if cluster_method is not None:
        config.cluster.method = cluster_method

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Loading sample from {sample_file}")
    sample = parakeet.sample.load(sample_file)

    # The exit wave file
    _exit_wave_Config(config, sample, exit_wave_file)


@exit_wave.register
def _exit_wave_Config(
    config: parakeet.config.Config, sample: parakeet.sample.Sample, exit_wave_file: str
):
    """
    Simulate the exit wave from the sample

    Args:
        config: The config object
        sample: The sample object
        exit_wave_file: The exit wave filename

    """

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Create the scan
    if config.scan.step_pos == "auto":
        radius = sample.shape_radius
        config.scan.step_pos = config.scan.step_angle * radius * pi / 180.0
    scan = parakeet.scan.new(**config.scan.dict())

    # Create the simulation
    simulation = simulation_factory(
        microscope,
        sample,
        scan,
        device=config.device,
        simulation=config.simulation.dict(),
        cluster=config.cluster.dict(),
    )

    # Create the writer
    logger.info(f"Opening file: {exit_wave_file}")
    writer = parakeet.io.new(
        exit_wave_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.complex64,
    )

    # Run the simulation
    simulation.simulate_image.metadata = writer.header
    simulation.run(writer)
