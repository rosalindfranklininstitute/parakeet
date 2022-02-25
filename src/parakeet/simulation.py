#
# parakeet.simulation.py
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
import mrcfile
import numpy as np
import warnings
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.sample
from math import sqrt, pi, sin, floor
from collections.abc import Iterable

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


def create_system_configuration(device):
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
    # source spread function
    ssf_sigma = multem.mrad_to_sigma(
        input_multislice.E_0, microscope.beam.source_spread
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

    def __init__(
        self, image_size, pixel_size, scan=None, cluster=None, simulate_image=None
    ):
        """
        Initialise the simulation

        Args:
            image_size (tuple): The image size
            scan (object): The scan object
            cluster (object): The cluster spec
            simulate_image (func): The image simulation function

        """
        self.pixel_size = pixel_size
        self.image_size = image_size
        self.scan = scan
        from .scan import UniformAngularScan

        if isinstance(scan, UniformAngularScan):  # single particle mode check
            self.scan.poses.write_star_file(self.scan.metadata_file)
        self.cluster = cluster
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
            return [0]
        return self.scan.angles

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
        if self.cluster is None or self.cluster["method"] is None:
            for i, angle in enumerate(self.angles()):
                logger.info(
                    f"    Running job: {i+1}/{self.shape[0]} for {angle} degrees"
                )
                _, angle, position, image, drift, defocus = self.simulate_image(i)
                if writer is not None:
                    writer.data[i, :, :] = image
                    writer.angle[i] = angle
                    writer.position[i] = position
                    if drift is not None:
                        writer.drift[i] = drift
                    if defocus is not None:
                        writer.defocus[i] = defocus
        else:

            # Set the maximum number of workers
            self.cluster["max_workers"] = min(
                self.cluster["max_workers"], self.shape[0]
            )
            logger.info("Initialising %d worker threads" % self.cluster["max_workers"])

            # Get the futures executor
            with parakeet.futures.factory(**self.cluster) as executor:

                # Copy the data to each worker
                logger.info("Copying data to workers...")

                # Submit all jobs
                logger.info("Running simulation...")
                futures = []
                for i, angle in enumerate(self.scan.angles):
                    logger.info(
                        f"    Submitting job: {i+1}/{self.shape[0]} for {angle} degrees"
                    )
                    futures.append(executor.submit(self.simulate_image, i))

                # Wait for results
                for j, future in enumerate(parakeet.futures.as_completed(futures)):

                    # Get the result
                    i, angle, position, image, drift, defocus = future.result()

                    # Set the output in the writer
                    if writer is not None:
                        writer.data[i, :, :] = image
                        writer.angle[i] = angle
                        writer.position[i] = position
                        if drift is not None:
                            writer.drift[i] = drift
                        if defocus is not None:
                            writer.defocus[i] = defocus

                    # Write some info
                    vmin = np.min(image)
                    vmax = np.max(image)
                    logger.info(
                        "    Processed job: %d (%d/%d); image min/max: %.2f/%.2f"
                        % (i + 1, j + 1, self.shape[0], vmin, vmax)
                    )


class ProjectedPotentialSimulator(object):
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
        position = self.scan.positions[index]

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

        # Create the multem system configuration
        system_conf = create_system_configuration(self.device)

        # The Z centre
        z_centre = self.sample.centre[2]

        # Create the multem input multislice object
        input_multislice = create_input_multislice(
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "EWRS",
            z_centre,
        )

        # Set the specimen size
        input_multislice.spec_lx = x_fov + offset * 2
        input_multislice.spec_ly = y_fov + offset * 2
        input_multislice.spec_lz = self.sample.containing_box[1][2]

        # Set the atoms in the input after translating them for the offset
        atoms = self.sample.get_atoms_in_fov(x0, x1)
        logger.info("Simulating with %d atoms" % atoms.data.shape[0])

        # Set atom sigma
        # atoms.data["sigma"] = sigma_B

        if len(atoms.data) > 0:
            coords = atoms.data[["x", "y", "z"]].to_numpy()
            coords = (
                self.scan.poses.orientations[index].apply(coords - self.sample.centre)
                + self.sample.centre
                - self.scan.poses.shifts[index]
            ).astype("float32")
            atoms.data["x"] = coords[:, 0]
            atoms.data["y"] = coords[:, 1]
            atoms.data["z"] = coords[:, 2]

        origin = (0, 0)
        input_multislice.spec_atoms = atoms.translate(
            (offset - origin[0], offset - origin[1], 0)
        ).to_multem()
        logger.info("   Got spec atoms")

        # Get the potential and thickness
        volume_z0 = self.sample.shape_box[0][2]
        volume_z1 = self.sample.shape_box[1][2]
        slice_thickness = self.simulation["slice_thickness"]
        zsize = int(floor((volume_z1 - volume_z0) / slice_thickness))
        potential = mrcfile.new_mmap(
            "projected_potential_%d.mrc" % index,
            shape=(zsize, ny, nx),
            mrc_mode=mrcfile.utils.mode_from_dtype(np.dtype(np.float32)),
            overwrite=True,
        )
        potential.voxel_size = tuple((pixel_size, pixel_size, slice_thickness))

        def callback(z0, z1, V):
            V = np.array(V)
            zc = (z0 + z1) / 2.0
            index = int(floor((zc - volume_z0) / slice_thickness))
            print(
                "Calculating potential for slice: %.2f -> %.2f (index: %d)"
                % (z0, z1, index)
            )
            potential.data[index, :, :] = V[margin:-margin, margin:-margin].T

        # Run the simulation
        multem.compute_projected_potential(system_conf, input_multislice, callback)

        # Compute the image scaled with Poisson noise
        return (index, angle, position, None, None, None)


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
        margin_offset = margin * pixel_size
        # padding_offset = padding * pixel_size
        offset = (padding + margin) * pixel_size

        # Create the multem system configuration
        system_conf = create_system_configuration(self.device)

        # The Z centre
        z_centre = self.sample.centre[2]

        # Create the multem input multislice object
        input_multislice = create_input_multislice(
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

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image, (driftx, drifty), None)


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
            system_conf = create_system_configuration(device)

            # Set the defocus
            if defocus is not None:
                microscope.lens.c_10 = defocus

            # Create the multem input multislice object
            input_multislice = create_input_multislice(
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


class ImageSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self, microscope=None, optics=None, scan=None, simulation=None, device="gpu"
    ):
        self.microscope = microscope
        self.optics = optics
        self.scan = scan
        self.simulation = simulation
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

        # Get the rotation angle
        angle = self.scan.angles[index]
        position = self.scan.positions[index]

        # Check the angle and position
        assert abs(angle - self.optics.angle[index]) < 1e7
        assert (np.abs(position - self.optics.position[index]) < 1e7).all()

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Compute the number of counts per pixel
        electrons_per_pixel = (
            self.microscope.beam.electrons_per_angstrom
            * self.microscope.detector.pixel_size**2
        )

        # Compute the electrons per pixel second
        electrons_per_second = electrons_per_pixel / self.scan.exposure_time
        energy = self.microscope.beam.energy

        # Get the image
        image = self.optics.data[index]

        # Get some other properties to propagate
        beam_drift = self.optics.drift[index]
        defocus = self.optics.defocus[index]

        # Apply the dqe in Fourier space
        if self.microscope.detector.dqe:
            logger.info("Applying DQE")
            dqe = parakeet.dqe.DQETable().dqe_fs(
                energy, electrons_per_second, image.shape
            )
            dqe = np.fft.fftshift(dqe)
            fft_image = np.fft.fft2(image)
            fft_image *= dqe
            image = np.real(np.fft.ifft2(fft_image))

        # Ensure all pixels are >= 0
        image = np.clip(image, 0, None)

        # Add Poisson noise
        # np.random.seed(index)
        image = np.random.poisson(image * electrons_per_pixel).astype("float64")

        # Print some info
        logger.info(
            "    Image min/max/mean: %g/%g/%.2g"
            % (np.min(image), np.max(image), np.mean(image))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image.astype("float32"), beam_drift, defocus)


class CTFSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(self, microscope=None, simulation=None):
        self.microscope = microscope
        self.simulation = simulation

    def __call__(self, index):
        """
        Simulate a single frame

        Args:
            simulation (object): The simulation object
            index (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size

        # Create the multem system configuration
        system_conf = create_system_configuration("cpu")

        # Create the multem input multislice object
        input_multislice = create_input_multislice(
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "HRTEM",
        )
        input_multislice.nx = nx
        input_multislice.ny = ny

        # Set the specimen size
        input_multislice.spec_lx = x_fov
        input_multislice.spec_ly = y_fov
        input_multislice.spec_lz = x_fov  # self.sample.containing_box[1][2]

        # Run the simulation
        image = np.array(multem.compute_ctf(system_conf, input_multislice)).T
        image = np.fft.fftshift(image)

        # Compute the image scaled with Poisson noise
        return (index, 0, 0, image, None, None)


class SimpleImageSimulator(object):
    """
    A class to do the actual simulation

    """

    def __init__(
        self, microscope=None, atoms=None, scan=None, simulation=None, device="gpu"
    ):
        self.microscope = microscope
        self.atoms = atoms
        self.scan = scan
        self.simulation = simulation
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

        # Get the rotation angle
        angle = self.scan.angles[index]
        position = self.scan.positions[index]

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

        # Set the rotation angle
        # input_multislice.spec_rot_theta = angle
        # input_multislice.spec_rot_u0 = simulation.scan.axis

        # x0 = (-offset, -offset)
        # x1 = (x_fov + offset, y_fov + offset)

        # Create the multem system configuration
        system_conf = create_system_configuration(self.device)

        # Create the multem input multislice object
        input_multislice = create_input_multislice(
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "EWRS",
        )

        # Set the specimen size
        input_multislice.spec_lx = x_fov + offset * 2
        input_multislice.spec_ly = y_fov + offset * 2
        input_multislice.spec_lz = np.max(self.atoms.data["z"])

        # Set the atoms in the input after translating them for the offset
        input_multislice.spec_atoms = self.atoms.translate(
            (offset, offset, 0)
        ).to_multem()

        # Run the simulation
        output_multislice = multem.simulate(system_conf, input_multislice)

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        image = np.array(output_multislice.data[0].psi_coh).T

        # Print some info
        psi_tot = np.abs(image) ** 2
        logger.info("Ideal image min/max: %f/%f" % (np.min(psi_tot), np.max(psi_tot)))

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image, None, None)


def projected_potential(
    microscope=None, sample=None, scan=None, device="gpu", simulation=None, cluster=None
):
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

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * simulation["margin"],
            microscope.detector.ny + 2 * simulation["margin"],
        ),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        cluster=cluster,
        simulate_image=ProjectedPotentialSimulator(
            microscope=microscope,
            sample=sample,
            scan=scan,
            simulation=simulation,
            device=device,
        ),
    )


def exit_wave(
    microscope=None, sample=None, scan=None, device="gpu", simulation=None, cluster=None
):
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

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * simulation["margin"],
            microscope.detector.ny + 2 * simulation["margin"],
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


def optics(
    microscope=None,
    exit_wave=None,
    scan=None,
    device="gpu",
    simulation=None,
    sample=None,
    cluster=None,
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


def image(
    microscope=None, optics=None, scan=None, device="gpu", simulation=None, cluster=None
):
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        optics (object): The optics object
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
        simulate_image=ImageSimulator(
            microscope=microscope,
            optics=optics,
            scan=scan,
            simulation=simulation,
            device=device,
        ),
    )


def simple(microscope=None, atoms=None, device="gpu", simulation=None):
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        atoms (object): The atom data
        device (str): The device to use
        simulation (object): The simulation parameters
        cluster (object): The cluster parameters

    Returns:
        object: The simulation object

    """
    scan = parakeet.scan.new("still")

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * simulation["margin"],
            microscope.detector.ny + 2 * simulation["margin"],
        ),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        cluster={"method": None},
        simulate_image=SimpleImageSimulator(
            microscope=microscope,
            scan=scan,
            atoms=atoms,
            simulation=simulation,
            device=device,
        ),
    )


def ctf(microscope=None, simulation=None):
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
        simulate_image=CTFSimulator(microscope=microscope, simulation=simulation),
    )
