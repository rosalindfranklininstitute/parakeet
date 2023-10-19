#
# parakeet.simulate.simulation.py
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
import warnings
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.sample
from math import sqrt, pi

# Try to input MULTEM
try:
    import multem
except ImportError:
    warnings.warn("Could not import MULTEM")


# Get the logger
logger = logging.getLogger(__name__)


def defocus_spread(Cc, dEE, dII, dVV):
    """
    From equation 3.41 in Kirkland: Advanced Computing in Electron Microscopy

    The dE, dI, dV are the 1/e half widths or E, I and V respectively

    Args:
        Cc (float): The chromatic abberation
        dEE (float): dE/E, the fluctuation in the electron energy
        dII (float): dI/I, the fluctuation in the lens current
        dVV (float): dV/V, the fluctuation in the acceleration voltage

    Returns:

    """
    return Cc * sqrt((dEE) ** 2 + (2 * dII) ** 2 + (dVV) ** 2)


def create_system_configuration(device, gpu_id=0):
    """
    Create an appropriate system configuration

    Args:
        device (str): The device to use

    Returns:
        object: The system configuration

    """
    assert device in ["cpu", "gpu"]

    # Initialise the system configuration
    system_conf = multem.SystemConfiguration()

    # Set the precision
    system_conf.precision = "float"

    # Set the device
    if device == "gpu":
        if multem.is_gpu_available():
            system_conf.device = "device"
        else:
            system_conf.device = "host"
            warnings.warn("GPU not present, reverting to CPU")
    else:
        system_conf.device = "host"

    # Set the gpu_device
    if gpu_id is not None:
        system_conf.gpu_device = gpu_id

    # Print some output
    logger.info("Simulating using %s" % system_conf.device)

    # Return the system configuration
    return system_conf


def create_input_multislice(
    microscope, slice_thickness, margin, simulation_type, centre=None
):
    """
    Create the input multislice object

    Args:
        microscope (object): The microscope object
        slice_thickness (float): The slice thickness
        margin (int): The pixel margin

    Returns:
        object: The input multislice object

    """

    # Initialise the input and system configuration
    input_multislice = multem.Input()

    # Set simulation experiment
    input_multislice.simulation_type = simulation_type

    # Electron-Specimen interaction model
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_type = "Lobato_0_12"

    # Potential slicing
    # XXX If this is set to "Planes" then for the ribosome example I found that
    # the simulation would not work well (e.g. The image may have nothing or a
    # single point of intensity and nothing else). Best to keep this set to
    # dz_Proj.
    input_multislice.potential_slicing = "dz_Proj"

    # Electron-Phonon interaction model
    input_multislice.pn_model = "Still_Atom"  # "Frozen_Phonon"
    # input_multislice.pn_model = "Frozen_Phonon"
    input_multislice.pn_coh_contrib = 0
    input_multislice.pn_single_conf = False
    input_multislice.pn_nconf = 50
    input_multislice.pn_dim = 110
    input_multislice.pn_seed = 300_183

    # Set the slice thickness
    input_multislice.spec_dz = slice_thickness

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"

    # x-y sampling
    input_multislice.nx = microscope.detector.nx + margin * 2
    input_multislice.ny = microscope.detector.ny + margin * 2
    input_multislice.bwl = False

    # Microscope parameters
    input_multislice.E_0 = microscope.beam.energy
    input_multislice.theta = microscope.beam.theta
    input_multislice.phi = microscope.beam.phi

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    # Condenser lens
    # source spread (illumination semiangle) function
    ssf_sigma = multem.mrad_to_sigma(
        input_multislice.E_0, microscope.beam.illumination_semiangle
    )
    input_multislice.cond_lens_si_sigma = ssf_sigma

    # Objective lens
    input_multislice.obj_lens_m = microscope.lens.m
    input_multislice.obj_lens_c_10 = microscope.lens.c_10
    input_multislice.obj_lens_c_12 = microscope.lens.c_12
    input_multislice.obj_lens_phi_12 = microscope.lens.phi_12
    input_multislice.obj_lens_c_21 = microscope.lens.c_21
    input_multislice.obj_lens_phi_21 = microscope.lens.phi_21
    input_multislice.obj_lens_c_23 = microscope.lens.c_23
    input_multislice.obj_lens_phi_23 = microscope.lens.phi_23
    input_multislice.obj_lens_c_30 = microscope.lens.c_30
    input_multislice.obj_lens_c_32 = microscope.lens.c_32
    input_multislice.obj_lens_phi_32 = microscope.lens.phi_32
    input_multislice.obj_lens_c_34 = microscope.lens.c_34
    input_multislice.obj_lens_phi_34 = microscope.lens.phi_34
    input_multislice.obj_lens_c_41 = microscope.lens.c_41
    input_multislice.obj_lens_phi_41 = microscope.lens.phi_41
    input_multislice.obj_lens_c_43 = microscope.lens.c_43
    input_multislice.obj_lens_phi_43 = microscope.lens.phi_43
    input_multislice.obj_lens_c_45 = microscope.lens.c_45
    input_multislice.obj_lens_phi_45 = microscope.lens.phi_45
    input_multislice.obj_lens_c_50 = microscope.lens.c_50
    input_multislice.obj_lens_c_52 = microscope.lens.c_52
    input_multislice.obj_lens_phi_52 = microscope.lens.phi_52
    input_multislice.obj_lens_c_54 = microscope.lens.c_54
    input_multislice.obj_lens_phi_54 = microscope.lens.phi_54
    input_multislice.obj_lens_c_56 = microscope.lens.c_56
    input_multislice.obj_lens_phi_56 = microscope.lens.phi_56
    input_multislice.obj_lens_inner_aper_ang = microscope.lens.inner_aper_ang
    input_multislice.obj_lens_outer_aper_ang = microscope.lens.outer_aper_ang

    # Do we have a phase plate
    # if microscope.phase_plate:
    #     input_multislice.phase_shift = pi / 2.0

    # defocus spread function
    input_multislice.obj_lens_ti_sigma = multem.iehwgd_to_sigma(
        defocus_spread(
            microscope.lens.c_c * 1e-3 / 1e-10,  # Convert from mm to A
            microscope.beam.energy_spread,
            microscope.lens.current_spread,
            microscope.beam.acceleration_voltage_spread,
        )
    )

    # zero defocus reference
    if centre is not None:
        input_multislice.cond_lens_zero_defocus_type = "User_Define"
        input_multislice.obj_lens_zero_defocus_type = "User_Define"
        input_multislice.cond_lens_zero_defocus_plane = centre
        input_multislice.obj_lens_zero_defocus_plane = centre
    else:
        input_multislice.cond_lens_zero_defocus_type = "Last"
        input_multislice.obj_lens_zero_defocus_type = "Last"

    # Return the input multislice object
    return input_multislice


def create_input_multislice_diffraction(
    microscope, slice_thickness, margin, simulation_type, centre=None
):
    """
    Create the input multislice object

    Args:
        microscope (object): The microscope object
        slice_thickness (float): The slice thickness
        margin (int): The pixel margin

    Returns:
        object: The input multislice object

    """

    # Initialise the input and system configuration
    input_multislice = multem.Input()

    # Set simulation experiment
    input_multislice.simulation_type = simulation_type

    # Electron-Specimen interaction model
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_type = "Lobato_0_12"

    # Potential slicing
    # XXX If this is set to "Planes" then for the ribosome example I found that
    # the simulation would not work well (e.g. The image may have nothing or a
    # single point of intensity and nothing else). Best to keep this set to
    # dz_Proj.
    input_multislice.potential_slicing = "dz_Proj"

    # Electron-Phonon interaction model
    input_multislice.pn_model = "Still_Atom"  # "Frozen_Phonon"
    # input_multislice.pn_model = "Frozen_Phonon"
    input_multislice.pn_coh_contrib = 0
    input_multislice.pn_single_conf = False
    input_multislice.pn_nconf = 50
    input_multislice.pn_dim = 110
    input_multislice.pn_seed = 300_183

    # Set the slice thickness
    input_multislice.spec_dz = slice_thickness

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"

    # x-y sampling
    input_multislice.nx = microscope.detector.nx + margin * 2
    input_multislice.ny = microscope.detector.ny + margin * 2
    input_multislice.bwl = False

    # Microscope parameters
    input_multislice.E_0 = microscope.beam.energy
    input_multislice.theta = microscope.beam.theta
    input_multislice.phi = microscope.beam.phi

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    # Set the incident wave
    # For some reason need this to work with CBED
    input_multislice.iw_x = [0]  # input_multislice.spec_lx/2
    input_multislice.iw_y = [0]  # input_multislice.spec_ly/2

    # Condenser lens
    # source spread (illumination semiangle) function
    ssf_sigma = multem.mrad_to_sigma(
        input_multislice.E_0, microscope.beam.illumination_semiangle
    )
    input_multislice.cond_lens_si_sigma = ssf_sigma

    # Objective lens
    input_multislice.cond_lens_m = microscope.lens.m
    input_multislice.cond_lens_c_10 = microscope.lens.c_10
    input_multislice.cond_lens_c_12 = microscope.lens.c_12
    input_multislice.cond_lens_phi_12 = microscope.lens.phi_12
    input_multislice.cond_lens_c_21 = microscope.lens.c_21
    input_multislice.cond_lens_phi_21 = microscope.lens.phi_21
    input_multislice.cond_lens_c_23 = microscope.lens.c_23
    input_multislice.cond_lens_phi_23 = microscope.lens.phi_23
    input_multislice.cond_lens_c_30 = microscope.lens.c_30
    input_multislice.cond_lens_c_32 = microscope.lens.c_32
    input_multislice.cond_lens_phi_32 = microscope.lens.phi_32
    input_multislice.cond_lens_c_34 = microscope.lens.c_34
    input_multislice.cond_lens_phi_34 = microscope.lens.phi_34
    input_multislice.cond_lens_c_41 = microscope.lens.c_41
    input_multislice.cond_lens_phi_41 = microscope.lens.phi_41
    input_multislice.cond_lens_c_43 = microscope.lens.c_43
    input_multislice.cond_lens_phi_43 = microscope.lens.phi_43
    input_multislice.cond_lens_c_45 = microscope.lens.c_45
    input_multislice.cond_lens_phi_45 = microscope.lens.phi_45
    input_multislice.cond_lens_c_50 = microscope.lens.c_50
    input_multislice.cond_lens_c_52 = microscope.lens.c_52
    input_multislice.cond_lens_phi_52 = microscope.lens.phi_52
    input_multislice.cond_lens_c_54 = microscope.lens.c_54
    input_multislice.cond_lens_phi_54 = microscope.lens.phi_54
    input_multislice.cond_lens_c_56 = microscope.lens.c_56
    input_multislice.cond_lens_phi_56 = microscope.lens.phi_56
    input_multislice.cond_lens_inner_aper_ang = microscope.lens.inner_aper_ang
    input_multislice.cond_lens_outer_aper_ang = microscope.lens.outer_aper_ang

    # Do we have a phase plate
    if microscope.phase_plate:
        input_multislice.phase_shift = pi / 2.0

    # defocus spread function
    input_multislice.obj_lens_ti_sigma = multem.iehwgd_to_sigma(
        defocus_spread(
            microscope.lens.c_c * 1e-3 / 1e-10,  # Convert from mm to A
            microscope.beam.energy_spread,
            microscope.lens.current_spread,
            microscope.beam.acceleration_voltage_spread,
        )
    )

    # zero defocus reference
    if centre is not None:
        input_multislice.cond_lens_zero_defocus_type = "User_Define"
        input_multislice.obj_lens_zero_defocus_type = "User_Define"
        input_multislice.cond_lens_zero_defocus_plane = centre
        input_multislice.obj_lens_zero_defocus_plane = centre
    else:
        input_multislice.cond_lens_zero_defocus_type = "Last"
        input_multislice.obj_lens_zero_defocus_type = "Last"

    # Return the input multislice object
    return input_multislice


class Simulation(object):
    """
    An object to wrap the simulation

    """

    def __init__(self, image_size, pixel_size, scan=None, nproc=1, simulate_image=None):
        """
        Initialise the simulation

        Args:
            image_size (tuple): The image size
            scan (object): The scan object
            nproc: The number of processes
            simulate_image (func): The image simulation function

        """
        self.pixel_size = pixel_size
        self.image_size = image_size
        self.scan = scan
        self.nproc = nproc
        self.simulate_image = simulate_image

    @property
    def shape(self):
        """
        Return
            tuple: The simulation data shape

        """
        nx = self.image_size[0]
        ny = self.image_size[1]
        nz = 1
        if self.scan is not None:
            nz = len(self.scan)
        return (nz, ny, nx)

    def angles(self):
        if self.scan is None:
            return [(0, 0, 0)]
        return list(
            zip(self.scan.image_number, self.scan.fraction_number, self.scan.angles)
        )

    def run(self, writer=None):
        """
        Run the simulation

        Args:
            writer (object): Write each image to disk

        """

        # Check the shape of the writer
        if writer:
            assert writer.shape == self.shape

        # If we are executing in a single process just do a for loop
        if self.nproc is None or self.nproc <= 1:
            for i, (image_number, fraction_number, angle) in enumerate(self.angles()):
                logger.info(
                    f"    Running job: {i+1}/{self.shape[0]} for image {image_number} fraction {fraction_number} with tilt {angle} degrees"
                )
                _, image, metadata = self.simulate_image(i)
                if writer is not None:
                    writer.data[i, :, :] = image
                    if metadata is not None:
                        writer.header[i] = metadata
        else:
            # Set the maximum number of workers
            self.nproc = min(self.nproc, self.shape[0])
            logger.info("Initialising %d worker threads" % self.nproc)

            # Get the futures executor
            with parakeet.futures.factory(max_workers=self.nproc) as executor:
                # Submit all jobs
                futures = []
                for i, (image_number, fraction_number, angle) in enumerate(
                    self.angles()
                ):
                    logger.info(
                        f"    Running job: {i+1}/{self.shape[0]} for image {image_number} fraction {fraction_number} with tilt {angle} degrees"
                    )
                    futures.append(executor.submit(self.simulate_image, i))

                # Wait for results
                for j, future in enumerate(parakeet.futures.as_completed(futures)):
                    # Get the result
                    i, image, metadata = future.result()

                    # Set the output in the writer
                    if writer is not None:
                        writer.data[i, :, :] = image
                        if metadata is not None:
                            writer.header[i] = metadata

                    # Write some info
                    vmin = np.min(image)
                    vmax = np.max(image)
                    logger.info(
                        "    Processed job: %d (%d/%d); image min/max: %.2f/%.2f"
                        % (i + 1, j + 1, self.shape[0], vmin, vmax)
                    )
