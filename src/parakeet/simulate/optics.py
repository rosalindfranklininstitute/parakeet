#
# parakeet.simulate.optics.py
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
import numpy as np
import warnings
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.io
import parakeet.sample
from parakeet.microscope import Microscope
from parakeet.scan import Scan
from functools import singledispatch
from math import sqrt, pi, sin
from parakeet.simulate.simulation import Simulation
from parakeet.config import Device
from parakeet.config import ClusterMethod

# Try to input MULTEM
try:
    import multem
except ImportError:
    warnings.warn("Could not import MULTEM")


# Get the logger
logger = logging.getLogger(__name__)


class OpticsImageSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self,
        microscope=None,
        exit_wave=None,
        scan=None,
        simulation=None,
        sample=None,
        device="gpu",
    ):
        self.microscope = microscope
        self.exit_wave = exit_wave
        self.scan = scan
        self.simulation = simulation
        self.sample = sample
        self.device = device

    def __call__(self, index):
        """
        Simulate a single frame

        Args:
            simulation (object): The simulation object
            index (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        def get_defocus(index, angle):
            """
            Get the defocus

            Returns:
                float: defocus
            """

            defocus = self.microscope.lens.c_10
            drift = 0
            defocus_drift = self.microscope.beam.defocus_drift
            if defocus_drift and not defocus_drift["type"] is None:
                if defocus_drift["type"] == "random":
                    drift = np.random.normal(0, defocus_drift["magnitude"])
                elif defocus_drift["type"] == "random_smoothed":
                    if index == 0:

                        def generate_smoothed_random(magnitude, num_images):
                            drift = np.random.normal(0, magnitude, size=(num_images))
                            drift = np.convolve(drift, np.ones(5) / 5, mode="same")
                            return drift

                        self._defocus_drift = generate_smoothed_random(
                            defocus_drift["magnitude"], len(self.scan)
                        )
                    drift = self._defocus_drift[index]
                elif defocus_drift["type"] == "sinusoidal":
                    drift = sin(angle * pi / 180) * defocus_drift["magnitude"]
                else:
                    raise RuntimeError("Unknown drift type")
                logger.info("Adding defocus drift of %f" % (drift))

            # Return the defocus
            return defocus + drift

        def compute_image(
            psi, microscope, simulation, x_fov, y_fov, offset, device, defocus=None
        ):

            # Create the multem system configuration
            system_conf = parakeet.simulate.simulation.create_system_configuration(
                device
            )

            # Set the defocus
            if defocus is not None:
                microscope.lens.c_10 = defocus

            # Create the multem input multislice object
            input_multislice = parakeet.simulate.simulation.create_input_multislice(
                microscope, simulation["slice_thickness"], simulation["margin"], "HRTEM"
            )

            # Set the specimen size
            input_multislice.spec_lx = x_fov + offset * 2
            input_multislice.spec_ly = y_fov + offset * 2
            input_multislice.spec_lz = x_fov  # self.sample.containing_box[1][2]

            # Compute and apply the CTF
            ctf = np.array(multem.compute_ctf(system_conf, input_multislice)).T

            # Compute the B factor for radiation damage
            # if simulation["radiation_damage_model"]:
            #    sigma_B = sqrt(
            #        simulation["sensitivity_coefficient"]
            #        * microscope.beam.electrons_per_angstrom
            #        * (index + 1)
            #    )
            #    pixel_size = microscope.detector.pixel_size
            #    Y, X = np.mgrid[0 : ctf.shape[0], 0 : ctf.shape[1]]
            #    X = (X - ctf.shape[1] // 2) / (pixel_size * ctf.shape[1])
            #    Y = (Y - ctf.shape[0] // 2) / (pixel_size * ctf.shape[0])
            #    q = np.sqrt(X ** 2 + Y ** 2)
            #    b_factor_blur = np.exp(-2 * pi ** 2 * q ** 2 * sigma_B ** 2)
            #    b_factor_blur = np.fft.fftshift(b_factor_blur)
            #    ctf = ctf * b_factor_blur

            # Compute and apply the CTF
            psi = np.fft.ifft2(np.fft.fft2(psi) * ctf)
            image = np.abs(psi) ** 2

            return image

        # Get the rotation angle
        angle = self.scan.angles[index]
        position = self.scan.positions[index]

        # Check the angle and position
        assert abs(angle - self.exit_wave.angle[index]) < 1e7
        assert (np.abs(position - self.exit_wave.position[index]) < 1e7).all()

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        margin = self.simulation["margin"]
        offset = margin * pixel_size

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Set the input wave
        psi = self.exit_wave.data[index]

        # Get the beam drift
        driftx, drifty = self.exit_wave.drift[index, :]

        microscope = copy.deepcopy(self.microscope)

        # Get the defocus
        defocus = get_defocus(index, angle)

        # If we do CC correction then set spherical aberration and chromatic
        # aberration to zero
        shape = self.sample["shape"]
        if self.simulation["inelastic_model"] is None:

            # If no inelastic model just calculate image as normal
            image = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )
            electron_fraction = 1.0

        elif self.simulation["inelastic_model"] == "zero_loss":

            # Compute the image
            image = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )

            # Calculate the fraction of electrons in the zero loss peak
            electron_fraction = parakeet.inelastic.zero_loss_fraction(shape, angle)

            # Scale the image by the fraction of electrons
            image *= electron_fraction

        elif self.simulation["inelastic_model"] == "mp_loss":

            # Set the filter width
            filter_width = self.simulation["mp_loss_width"]  # eV

            # Compute the energy and spread of the plasmon peak
            thickness = parakeet.inelastic.effective_thickness(shape, angle)  # A
            peak, sigma = parakeet.inelastic.most_probable_loss(
                microscope.beam.energy, shape, angle
            )  # eV

            # Save the energy and energy spread
            beam_energy = microscope.beam.energy * 1000  # eV
            beam_energy_spread = microscope.beam.energy_spread  # dE / E
            beam_energy_sigma = (1.0 / sqrt(2)) * beam_energy_spread * beam_energy  # eV

            # Set a maximum peak energy loss
            peak = min(peak, beam_energy * 0.1)  # eV

            # Make optimizer
            optimizer = parakeet.inelastic.EnergyFilterOptimizer(dE_min=-60, dE_max=200)
            assert self.simulation["mp_loss_position"] in ["peak", "optimal"]
            if self.simulation["mp_loss_position"] != "peak":
                peak = optimizer(beam_energy, thickness, filter_width=filter_width)

            # Compute elastic fraction and spread
            elastic_fraction, elastic_spread = optimizer.compute_elastic_component(
                beam_energy, thickness, peak, filter_width
            )

            # Compute inelastic fraction and spread
            (
                inelastic_fraction,
                inelastic_spread,
            ) = optimizer.compute_inelastic_component(
                beam_energy, thickness, peak, filter_width
            )

            # Compute the spread
            elastic_spread = elastic_spread / beam_energy  # dE / E
            inelastic_spread = inelastic_spread / beam_energy  # dE / E

            # Set the spread for the zero loss image
            microscope.beam.energy_spread = elastic_spread  # dE / E

            # Compute the zero loss image
            image1 = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )

            # Add the energy loss
            microscope.beam.energy = (beam_energy - peak) / 1000.0  # keV

            # Compute the energy spread of the plasmon peak
            microscope.beam.energy_spread = (
                beam_energy_spread + inelastic_spread
            )  # dE / E
            print("Energy: %f keV" % microscope.beam.energy)
            print("Energy spread: %f ppm" % microscope.beam.energy_spread)

            # Compute the MPL image
            image2 = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )

            # Compute the zero loss and mpl image fraction
            electron_fraction = elastic_fraction + inelastic_fraction

            # Add the images incoherently and scale the image by the fraction of electrons
            image = elastic_fraction * image1 + inelastic_fraction * image2

        elif self.simulation["inelastic_model"] == "unfiltered":

            # Compute the energy and spread of the plasmon peak
            peak, sigma = parakeet.inelastic.most_probable_loss(
                microscope.beam.energy, shape, angle
            )  # eV
            peak = min(peak, 1000 * microscope.beam.energy * 0.1)  # eV
            spread = sigma * sqrt(2) / (microscope.beam.energy * 1000)  # dE / E

            # Compute the zero loss image
            image1 = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )

            # Add the energy loss
            microscope.beam.energy -= peak / 1000  # keV

            # Compute the energy spread of the plasmon peak
            microscope.beam.energy_spread += spread  # dE / E
            print("Energy: %f keV" % microscope.beam.energy)
            print("Energy spread: %f ppm" % microscope.beam.energy_spread)

            # Compute the MPL image
            image2 = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )

            # Compute the zero loss and mpl image fraction
            zero_loss_fraction = parakeet.inelastic.zero_loss_fraction(shape, angle)
            mp_loss_fraction = parakeet.inelastic.mp_loss_fraction(shape, angle)
            electron_fraction = zero_loss_fraction + mp_loss_fraction

            # Add the images incoherently and scale the image by the fraction of electrons
            image = zero_loss_fraction * image1 + mp_loss_fraction * image2

        elif self.simulation["inelastic_model"] == "cc_corrected":

            # Set the Cs and CC to zero
            microscope.lens.c_30 = 0
            microscope.lens.c_c = 0

            # Compute the energy and spread of the plasmon peak
            peak, sigma = parakeet.inelastic.most_probable_loss(
                microscope.beam.energy, shape, angle
            )
            peak /= 1000.0
            peak = min(peak, microscope.beam.energy * 0.1)
            spread = sigma * sqrt(2) / (microscope.beam.energy * 1000)

            # Compute the zero loss image
            image1 = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )

            # Add the energy loss
            microscope.beam.energy -= peak

            # Compute the energy spread of the plasmon peak
            microscope.beam.energy_spread += spread
            print("Energy: %f keV" % microscope.beam.energy)
            print("Energy spread: %f ppm" % microscope.beam.energy_spread)

            # Compute the MPL image
            image2 = compute_image(
                psi,
                microscope,
                self.simulation,
                x_fov,
                y_fov,
                offset,
                self.device,
                defocus,
            )

            # Compute the zero loss and mpl image fraction
            zero_loss_fraction = parakeet.inelastic.zero_loss_fraction(shape, angle)
            mp_loss_fraction = parakeet.inelastic.mp_loss_fraction(shape, angle)
            electron_fraction = zero_loss_fraction + mp_loss_fraction

            # Add the images incoherently and scale the image by the fraction of electrons
            image = zero_loss_fraction * image1 + mp_loss_fraction * image2

        else:
            raise RuntimeError("Unknown inelastic model")

        # Print the electron fraction
        print("Electron fraction = %.2f" % electron_fraction)

        # Remove margin
        j0 = margin
        i0 = margin
        j1 = image.shape[0] - margin
        i1 = image.shape[1] - margin
        assert margin >= 0
        assert i1 > i0
        assert j1 > j0
        image = image[j0:j1, i0:i1]
        logger.info(
            "Ideal image min/mean/max: %f/%f/%f"
            % (np.min(image), np.mean(image), np.max(image))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image, (driftx, drifty), defocus)


def optics_internal(
    microscope: Microscope,
    exit_wave: object,
    scan: Scan,
    device: Device = Device.gpu,
    simulation: dict = None,
    sample: dict = None,
    cluster: dict = None,
):
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        exit_wave (object): The exit_wave object
        scan (object): The scan object
        device (str): The device to use
        simulation (object): The simulation parameters
        cluster (object): The cluster parameters

    Returns:
        object: The simulation object

    """

    # Create the simulation
    return Simulation(
        image_size=(microscope.detector.nx, microscope.detector.ny),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        cluster=cluster,
        simulate_image=OpticsImageSimulator(
            microscope=microscope,
            exit_wave=exit_wave,
            scan=scan,
            simulation=simulation,
            sample=sample,
            device=device,
        ),
    )


@singledispatch
def optics(
    config_file,
    exit_wave_file: str,
    optics_file: str,
    device: Device = Device.gpu,
    cluster_method: ClusterMethod = None,
    cluster_max_workers: int = 1,
):
    """
    Simulate the optics

    Args:
        config_file: The input config filename
        exit_wave_file: The input exit wave filename
        optics_file: The output optics filename
        device: The device to run on (CPU or GPU)
        cluster_method: The cluster method to use (default None)
        cluster_max_workers: The maximum number of cluster jobs

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the device in a dict
    if device is not None:
        config.device = device
    if cluster_max_workers is not None:
        config.cluster.max_workers = cluster_max_workers
    if cluster_method is not None:
        config.cluster.method = cluster_method

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the exit wave data
    logger.info(f"Loading sample from {exit_wave_file}")
    exit_wave = parakeet.io.open(exit_wave_file)

    # Create the scan
    scan = parakeet.scan.new(
        angles=exit_wave.angle, positions=exit_wave.position[:, 1], **config.scan.dict()
    )

    # Create the simulation
    simulation = optics_internal(
        microscope,
        exit_wave,
        scan,
        device=config.device,
        simulation=config.simulation.dict(),
        sample=config.sample.dict(),
        cluster=config.cluster.dict(),
    )

    # Create the writer
    logger.info(f"Opening file: {optics_file}")
    writer = parakeet.io.new(
        optics_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.float32,
    )

    # Run the simulation
    simulation.run(writer)


# Register function for single dispatch
optics.register(optics_internal)
