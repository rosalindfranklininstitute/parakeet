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
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.io
import parakeet.sample
from parakeet.config import Device
from parakeet.microscope import Microscope
from parakeet.scan import Scan
from functools import singledispatch
from parakeet.simulate.simulation import Simulation
from parakeet.simulate.engine import SimulationEngine
from parakeet.microscope import Microscope
from parakeet.scan import Scan


__all__ = ["optics"]


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
        gpu_id=None,
    ):
        self.microscope = microscope
        self.exit_wave = exit_wave
        self.scan = scan
        self.simulation = simulation
        self.sample = sample
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

        def compute_image(
            psi,
            microscope,
            simulation,
            x_fov,
            y_fov,
            offset,
            device,
            gpu_id,
            defocus=None,
        ):
            # Set the defocus
            if defocus is not None:
                microscope.lens.c_10 = defocus

            # Create the simulation engine
            simulate = SimulationEngine(
                device,
                gpu_id,
                microscope,
                simulation["slice_thickness"],
                simulation["margin"],
                "HRTEM",
            )

            # Set the specimen size
            simulate.input.spec_lx = x_fov + offset * 2
            simulate.input.spec_ly = y_fov + offset * 2
            simulate.input.spec_lz = x_fov  # self.sample.containing_box[1][2]

            # Compute and apply the CTF
            ctf = simulate.ctf()

            # Apply an objective aperture cutoff frequency
            if microscope.objective_aperture_cutoff_freq is not None:
                qy = np.fft.fftfreq(ctf.shape[0], d=pixel_size)
                qx = np.fft.fftfreq(ctf.shape[1], d=pixel_size)
                q = np.sqrt(qy[:, None] ** 2 + qx[None, :] ** 2)
                mask = q < microscope.objective_aperture_cutoff_freq
                ctf = ctf * mask

            # Add the effect of the phase plate
            if microscope.phase_plate.use:
                ctf = ctf * parakeet.simulate.phase_plate.compute_phase_shift(
                    ctf.shape,
                    microscope.detector.pixel_size,
                    microscope.phase_plate.phase_shift,
                    microscope.phase_plate.radius,
                )
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
        defocus_offset = self.scan.defocus_offset[index]

        # Check the angle and position
        assert abs(angle - self.exit_wave.header[index]["tilt_alpha"]) < 1e7

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

        microscope = copy.deepcopy(self.microscope)

        # Get the defocus
        defocus = microscope.lens.c_10 + defocus_offset

        # If we do CC correction then set spherical aberration and chromatic
        # aberration to zero
        shape = self.sample["shape"]
        energy_shift = 0
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
                self.gpu_id,
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
                self.gpu_id,
                defocus,
            )

            # Calculate the fraction of electrons in the zero loss peak
            electron_fraction = parakeet.inelastic.zero_loss_fraction(shape, angle)

            # Scale the image by the fraction of electrons
            image *= electron_fraction

        else:
            # Get the effective thickness
            thickness = parakeet.inelastic.effective_thickness(shape, angle)  # A
            if self.simulation["inelastic_model"] == "unfiltered":
                # Get the energy bins
                bin_energy, bin_spread, bin_weight = parakeet.inelastic.get_energy_bins(
                    energy=microscope.beam.energy * 1000,  # eV
                    thickness=thickness,
                    energy_spread=microscope.beam.energy_spread
                    * microscope.beam.energy
                    * 1000,  # dE
                )

            elif self.simulation["inelastic_model"] == "cc_corrected":
                # Get the energy bins
                bin_energy, bin_spread, bin_weight = parakeet.inelastic.get_energy_bins(
                    energy=microscope.beam.energy * 1000,  # eV
                    thickness=thickness,
                    energy_spread=microscope.beam.energy_spread
                    * microscope.beam.energy
                    * 1000,  # dE
                )

                # Set the Cs and CC to zero
                microscope.lens.c_30 = 0
                microscope.lens.c_c = 0

            elif self.simulation["inelastic_model"] == "mp_loss":
                # Set the filter width
                filter_width = self.simulation["mp_loss_width"]  # eV

                # Make optimizer
                optimizer = parakeet.inelastic.EnergyFilterOptimizer(
                    dE_min=-60, dE_max=200
                )
                assert self.simulation["mp_loss_position"] in ["peak", "optimal"]

                # Compute the energy and spread of the plasmon peak
                if self.simulation["mp_loss_position"] != "peak":
                    peak = optimizer(
                        microscope.beam.energy, thickness, filter_width=filter_width
                    )
                else:
                    peak, sigma = parakeet.inelastic.most_probable_loss(
                        microscope.beam.energy, shape, angle
                    )  # eV

                # Set a maximum peak energy loss at 10% of beam energy
                peak = min(peak, microscope.beam.energy * 1000 * 0.1)  # eV

                # Get the energy bins
                bin_energy, bin_spread, bin_weight = parakeet.inelastic.get_energy_bins(
                    energy=microscope.beam.energy * 1000,  # eV
                    thickness=thickness,
                    energy_spread=microscope.beam.energy_spread
                    * microscope.beam.energy
                    * 1000,  # dE
                    filter_energy=peak,
                    filter_width=filter_width,
                )

            else:
                raise RuntimeError("Unknown inelastic model")

            # Get the threshold to exclude bins that don't contribute much
            threshold = min(0.01 / len(bin_energy), max(bin_weight))
            print("Threshold weight: %f" % (threshold))

            # Select based on threshold
            selection = bin_weight >= threshold
            bin_energy = bin_energy[selection]
            bin_spread = bin_spread[selection]
            bin_weight = bin_weight[selection]

            # Get the basic energy and defocus
            energy0 = microscope.beam.energy
            defocus0 = microscope.lens.c_10

            # Energy and energy spread
            energy1 = bin_energy / 1000.0  # keV
            energy_spread1 = bin_spread / bin_energy  # dE / E

            # Compute the defocus at this point
            # Energy loss is positive.
            # Energy loss results in over focus which is also positive
            c_c_A = microscope.lens.c_c * 1e7  # A
            dE_E = (energy0 - energy1) / energy0
            defocus1 = defocus0 + c_c_A * dE_E  # A

            # Adjust defocus to mean
            # defocus_mean = np.average(defocus1, weights=bin_weight)
            # defocus1 = defocus1 + (defocus0 - defocus_mean)

            # Loop through all energies and sum images
            image = None
            for energy, energy_spread, defocus, weight in zip(
                energy1, energy_spread1, defocus1, bin_weight
            ):
                # Add the energy loss
                microscope.beam.energy = energy  # keV

                # Compute the energy spread
                microscope.beam.energy_spread = energy_spread  # dE / E

                # Print some details
                print(
                    "Energy: %f eV; Energy spread: %f eV; Weight: %f; Defocus: %f"
                    % (
                        microscope.beam.energy * 1000,
                        microscope.beam.energy_spread * microscope.beam.energy * 1000,
                        weight,
                        defocus,
                    )
                )

                # Compute the MPL image
                image_n = weight * compute_image(
                    psi,
                    microscope,
                    self.simulation,
                    x_fov,
                    y_fov,
                    offset,
                    self.device,
                    self.gpu_id,
                    defocus,
                )

                # Add image component
                if image is None:
                    image = image_n
                else:
                    image += image_n

            # Compute the electron fraction
            electron_fraction = np.sum(bin_weight)

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

        # Set the metadata
        metadata = np.asarray(self.exit_wave.header[index])
        metadata["c_10"] = defocus
        metadata["c_12"] = self.microscope.lens.c_12
        metadata["c_21"] = self.microscope.lens.c_21
        metadata["c_23"] = self.microscope.lens.c_23
        metadata["c_30"] = self.microscope.lens.c_30
        metadata["c_32"] = self.microscope.lens.c_32
        metadata["c_34"] = self.microscope.lens.c_34
        metadata["c_41"] = self.microscope.lens.c_41
        metadata["c_43"] = self.microscope.lens.c_43
        metadata["c_45"] = self.microscope.lens.c_45
        metadata["c_50"] = self.microscope.lens.c_50
        metadata["c_52"] = self.microscope.lens.c_52
        metadata["c_54"] = self.microscope.lens.c_54
        metadata["c_56"] = self.microscope.lens.c_56
        metadata["phi_12"] = self.microscope.lens.phi_12
        metadata["phi_21"] = self.microscope.lens.phi_21
        metadata["phi_23"] = self.microscope.lens.phi_23
        metadata["phi_32"] = self.microscope.lens.phi_32
        metadata["phi_34"] = self.microscope.lens.phi_34
        metadata["phi_41"] = self.microscope.lens.phi_41
        metadata["phi_43"] = self.microscope.lens.phi_43
        metadata["phi_45"] = self.microscope.lens.phi_45
        metadata["phi_52"] = self.microscope.lens.phi_52
        metadata["phi_54"] = self.microscope.lens.phi_54
        metadata["phi_56"] = self.microscope.lens.phi_56
        metadata["c_c"] = self.microscope.lens.c_c
        metadata["current_spread"] = self.microscope.lens.current_spread
        metadata["illumination_semiangle"] = self.microscope.beam.illumination_semiangle
        metadata["acceleration_voltage_spread"] = (
            self.microscope.beam.acceleration_voltage_spread
        )
        metadata["energy_spread"] = self.microscope.beam.energy_spread
        metadata["phase_plate"] = self.microscope.phase_plate
        metadata["inelastic_model"] = self.simulation["inelastic_model"]
        metadata["slit_inserted"] = self.simulation["inelastic_model"] in [
            "zero_loss",
            "mp_loss",
        ]
        metadata["slit_width"] = self.simulation["mp_loss_width"]
        metadata["energy_shift"] = energy_shift

        # Compute the image scaled with Poisson noise
        return (index, image, metadata)


def simulation_factory(
    microscope: Microscope,
    exit_wave: object,
    scan: Scan,
    simulation: dict = None,
    sample: dict = None,
    multiprocessing: dict = None,
) -> Simulation:
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        exit_wave (object): The exit_wave object
        scan (object): The scan object
        simulation (object): The simulation parameters
        multiprocessing (object): The multiprocessing parameters

    Returns:
        object: The simulation object

    """

    # Check multiprocessing settings
    if multiprocessing is None:
        multiprocessing = {"device": "gpu", "nproc": 1, "gpu_id": 0}
    else:
        assert multiprocessing["nproc"] in [None, 1]
        assert len(multiprocessing["gpu_id"]) == 1

    # Create the simulation
    return Simulation(
        image_size=(microscope.detector.nx, microscope.detector.ny),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        simulate_image=OpticsImageSimulator(
            microscope=microscope,
            exit_wave=exit_wave,
            scan=scan,
            simulation=simulation,
            sample=sample,
            device=multiprocessing["device"],
            gpu_id=multiprocessing["gpu_id"][0],
        ),
    )


@singledispatch
def optics(
    config_file,
    exit_wave_file: str,
    optics_file: str,
    device: Device = None,
    nproc: int = None,
    gpu_id: list = None,
):
    """
    Simulate the optics

    Args:
        config_file: The input config filename
        exit_wave_file: The input exit wave filename
        optics_file: The output optics filename
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
    _optics_Config(config, exit_wave_file, optics_file)


@optics.register(parakeet.config.Config)
def _optics_Config(
    config: parakeet.config.Config, exit_wave_file: str, optics_file: str
):
    """
    Simulate the optics

    Args:
        config: The input config
        exit_wave_file: The input exit wave filename
        optics_file: The output optics filename

    """

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Create the exit wave data
    logger.info(f"Loading sample from {exit_wave_file}")
    exit_wave = parakeet.io.open(exit_wave_file)

    # Create the scan
    scan = exit_wave.header.scan

    # Override the defocus_offset
    scan_new = parakeet.scan.new(**config.scan.model_dump())
    scan.data["defocus_offset"] = scan_new.defocus_offset

    # Create the simulation
    simulation = simulation_factory(
        microscope,
        exit_wave,
        scan,
        simulation=config.simulation.model_dump(),
        sample=config.sample.model_dump(),
        multiprocessing=config.multiprocessing.model_dump(),
    )

    # Create the writer
    logger.info(f"Opening file: {optics_file}")
    writer = parakeet.io.new(
        optics_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.float32,
    )

    # Propagate the particle positions
    writer.particle_positions = exit_wave.particle_positions

    # Run the simulation
    simulation.run(writer)
