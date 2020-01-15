#
# elfantasma.simulation.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#

import logging
import numpy
import warnings
import elfantasma.config
import elfantasma.freeze
import elfantasma.futures
import elfantasma.sample
import warnings
from math import sqrt

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


def create_input_multislice(microscope, slice_thickness, margin, simulation_type):
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
    input_multislice.pn_model = "Still_Atom"
    input_multislice.pn_coh_contrib = 0
    input_multislice.pn_single_conf = False
    input_multislice.pn_nconf = 10
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
    input_multislice.theta = 0.0
    input_multislice.phi = 0.0

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    # Condenser lens
    # source spread function
    # ssf_sigma = multem.mrad_to_sigma(input_multislice.E_0, 0.02)
    # input_multislice.obj_lens_ssf_sigma = ssf_sigma

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

    # defocus spread function
    input_multislice.obj_lens_dsf_sigma = defocus_spread(
        microscope.lens.c_c,
        microscope.beam.energy_spread,
        microscope.lens.current_spread,
        microscope.beam.acceleration_voltage_spread,
    )

    # zero defocus reference
    input_multislice.obj_lens_zero_defocus_type = "Last"

    # Return the input multislice object
    return input_multislice


class Simulation(object):
    """
    An object to wrap the simulation

    """

    def __init__(self, image_size, scan=None, cluster=None, simulate_image=None):
        """
        Initialise the simulation

        Args:
            image_size (tuple): The image size
            scan (object): The scan object
            cluster (object): The cluster spec
            simulate_image (func): The image simulation function

        """
        self.image_size = image_size
        self.scan = scan
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
        nz = len(self.scan)
        return (nz, ny, nx)

    def run(self, writer):
        """
        Run the simulation

        Args:
            writer (object): Write each image to disk

        """

        # Check the shape of the writer
        assert writer.shape == self.shape

        # If we are executing in a single process just do a for loop
        if self.cluster["method"] is None:
            for i, angle in enumerate(self.scan.angles):
                logger.info(
                    f"    Running job: {i+1}/{self.shape[0]} for {angle} degrees"
                )
                _, angle, image = self.simulate_image(i)
                writer.data[i, :, :] = image
                writer.angle[i] = angle
        else:

            # Set the maximum number of workers
            self.cluster["max_workers"] = min(
                self.cluster["max_workers"], self.shape[0]
            )
            logger.info("Initialising %d worker threads" % self.cluster["max_workers"])

            # Get the futures executor
            with elfantasma.futures.factory(**self.cluster) as executor:

                # Copy the data to each worker
                logger.info("Copying data to workers...")

                # Submit all jobs
                logger.info("Running simulation...")
                futures = []
                for i, angle in enumerate(self.scan.angles):
                    logger.info(
                        f"    Submitting job: {i+1}/{self.shape[0]} for {angle} degrees"
                    )
                    futures.append(executor.submit(simulate_image, i))

                # Wait for results
                for j, future in enumerate(elfantasma.futures.as_completed(futures)):

                    # Get the result
                    i, angle, image = future.result()

                    # Set the output in the writer
                    writer.data[i, :, :] = image
                    writer.angle[i] = angle

                    # Write some info
                    vmin = numpy.min(image)
                    vmax = numpy.max(image)
                    logger.info(
                        "    Processed job: %d (%d/%d); image min/max: %.2f/%.2f"
                        % (i + 1, j + 1, self.shape[0], vmin, vmax)
                    )


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
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        offset = self.simulation["margin"] * pixel_size

        # Get the specimen atoms
        logger.info(f"Simulating image {index}")

        # Set the rotation angle
        # input_multislice.spec_rot_theta = angle
        # input_multislice.spec_rot_u0 = simulation.scan.axis

        # Create the sample extractor
        x0 = (0, 0)
        x1 = (x_fov + position, y_fov)
        thickness = self.simulation["division_thickness"]
        extractor = elfantasma.sample.AtomSliceExtractor(
            self.sample, angle, position, x0, x1, thickness
        )

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
        input_multislice.spec_lz = self.sample.containing_box[1][2]

        # Either slice or don't
        if len(extractor) == 1:

            # Set the atoms in the input after translating them for the offset
            zslice = extractor[0]
            logger.info(
                "    Simulating z slice %f -> %f with %d atoms"
                % (zslice.x_min[2], zslice.x_max[2], zslice.atoms.data.shape[0])
            )
            input_multislice.spec_atoms = zslice.atoms.translate(
                (offset, offset, 0)
            ).to_multem()

            # Run the simulation
            output_multislice = multem.simulate(system_conf, input_multislice)

        else:

            # Slice the specimen atoms
            def slice_generator(extractor):
                for zslice in extractor:
                    logger.info(
                        "    Simulating z slice %f -> %f with %d atoms"
                        % (zslice.x_min[2], zslice.x_max[2], zslice.atoms.data.shape[0])
                    )
                    x = (
                        zslice.x_min[2],
                        zslice.x_max[2] - zslice.x_min[2],
                        zslice.atoms.translate((offset, offset, 0)).to_multem(),
                    )
                    yield x

            # Run the simulation
            output_multislice = multem.simulate(
                system_conf, input_multislice, slice_generator(extractor)
            )

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        image = numpy.array(output_multislice.data[0].psi_coh).T

        # Remove margin
        margin = self.simulation["margin"]
        j0 = margin
        i0 = margin
        j1 = image.shape[0] - margin
        i1 = image.shape[1] - margin
        assert margin >= 0
        assert i1 > i0
        assert j1 > j0
        image = image[j0:j1, i0:i1]
        psi_tot = numpy.abs(image) ** 2
        logger.info(
            "Ideal image min/max: %f/%f" % (numpy.min(psi_tot), numpy.max(psi_tot))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, image)


class OpticsImageSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self, microscope=None, exit_wave=None, scan=None, simulation=None, device="gpu"
    ):
        self.microscope = microscope
        self.exit_wave = exit_wave
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
        assert abs(angle - self.exit_wave.angle[index]) < 1e7
        assert (numpy.abs(position - self.exit_wave.position[index]) < 1e7).all()

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        offset = self.simulation["margin"] * pixel_size

        # Get the specimen atoms
        logger.info(f"Simulating image {index}")

        # Set the rotation angle
        # input_multislice.spec_rot_theta = angle
        # input_multislice.spec_rot_u0 = simulation.scan.axis

        # Create the sample extractor
        # x0 = (0, 0)
        # x1 = (x_fov + position, y_fov)
        # thickness = 10
        # extractor = elfantasma.sample.AtomSliceExtractor(
        #     self.sample, angle, position, x0, x1, thickness
        # )

        # Create the multem system configuration
        system_conf = create_system_configuration(self.device)

        # Create the multem input multislice object
        input_multislice = create_input_multislice(
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "HRTEM",
        )

        # Set the specimen size
        input_multislice.spec_lx = x_fov + offset * 2
        input_multislice.spec_ly = y_fov + offset * 2
        input_multislice.spec_lz = x_fov  # self.sample.containing_box[1][2]

        # Set the input wave
        margin = self.simulation["margin"]
        user_defined_wave = numpy.zeros(
            shape=(ny + 2 * margin, nx + 2 * margin), dtype=numpy.complex64
        )
        j0 = margin
        i0 = margin
        j1 = user_defined_wave.shape[0] - margin
        i1 = user_defined_wave.shape[1] - margin
        user_defined_wave[j0:j1, i0:i1] = self.exit_wave.data[index]
        input_multislice.iw_type = "User_Define_Wave"
        input_multislice.iw_psi = list(user_defined_wave.T.flatten())
        input_multislice.iw_x = [0.5 * input_multislice.spec_lx]
        input_multislice.iw_y = [0.5 * input_multislice.spec_ly]
        input_multislice.spec_atoms = multem.AtomList(
            [
                (
                    1,
                    input_multislice.spec_lx / 2.0,
                    input_multislice.spec_ly / 2.0,
                    input_multislice.spec_lz / 2.0,
                    0.088,
                    1,
                    0,
                    0,
                )
            ]
        )

        # Run the simulation
        output_multislice = multem.simulate(system_conf, input_multislice)

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        image = numpy.array(output_multislice.data[0].m2psi_tot).T

        # Remove margin
        j0 = margin
        i0 = margin
        j1 = image.shape[0] - margin
        i1 = image.shape[1] - margin
        assert margin >= 0
        assert i1 > i0
        assert j1 > j0
        image = image[j0:j1, i0:i1]
        psi_tot = numpy.abs(image) ** 2
        logger.info(
            "Ideal image min/max: %f/%f" % (numpy.min(psi_tot), numpy.max(psi_tot))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, image)


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
        assert (numpy.abs(position - self.optics.position[index]) < 1e7).all()

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size

        # Get the specimen atoms
        logger.info(f"Simulating image {index}")

        # Compute the number of counts per pixel
        N = (
            self.microscope.beam.electrons_per_angstrom
            / self.microscope.detector.pixel_size
        )

        # Compute the new image
        image = self.optics.data[index] / numpy.mean(self.optics.data[index])
        image = numpy.random.poisson(image * N)

        # Print some info
        logger.info(
            "    Image min/max/mean: %d/%d/%d"
            % (numpy.min(image), numpy.max(image), numpy.mean(image))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, image)


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
        image_size=(microscope.detector.nx, microscope.detector.ny),
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
        scan=scan,
        cluster=cluster,
        simulate_image=OpticsImageSimulator(
            microscope=microscope,
            exit_wave=exit_wave,
            scan=scan,
            simulation=simulation,
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


# class SingleImageSimulation(object):
#     """
#     A class to do the actual simulation

#     The simulation is structured this way because the input data to the
#     simulation is large enough that it makes an overhead to creating the
#     individual processes.

#     """

#     def __call__(self, simulation, i):
#         """
#         Simulate a single frame

#         Args:
#             simulation (object): The simulation object
#             i (int): The frame number

#         Returns:
#             tuple: (angle, image)

#         """

#         # Create the multem system configuration
#         system_conf = create_system_configuration(simulation.device)

#         # Create the multem input multislice object
#         input_multislice = create_input_multislice(
#             simulation.microscope,
#             simulation.slice_thickness,
#             simulation.margin,
#             simulation.simulation_type,
#         )

#         # Get the rotation angle
#         angle = simulation.scan.angles[i]
#         position = simulation.scan.positions[i]

#         # Set the rotation angle
#         input_multislice.spec_rot_theta = angle
#         input_multislice.spec_rot_u0 = simulation.scan.axis

#         # The field of view
#         x_fov = (
#             simulation.microscope.detector.nx
#             * simulation.microscope.detector.pixel_size
#         )
#         y_fov = (
#             simulation.microscope.detector.ny
#             * simulation.microscope.detector.pixel_size
#         )
#         offset = simulation.margin * simulation.microscope.detector.pixel_size

#         # Get the specimen atoms
#         logger.info(f"Simulating image {i}")
#         atom_data = self.atom_data(
#             simulation.sample, position, position + x_fov, 0, y_fov, offset
#         )

#         # Set the specimen size
#         input_multislice.spec_lx = x_fov + offset * 2
#         input_multislice.spec_ly = y_fov + offset * 2
#         input_multislice.spec_lz = simulation.sample.box_size[2]

#         # Either slice or don't
#         if simulation.num_slices == 1:

#             # Set the atoms in the input
#             input_multislice.spec_atoms = multem.AtomList(
#                 elfantasma.sample.extract_spec_atoms(atom_data)
#             )

#             # Run the simulation
#             output_multislice = multem.simulate(system_conf, input_multislice)

#         else:

#             # Slice the specimen atoms
#             spec_slices = list(
#                 self.slice_atom_data(
#                     atom_data, input_multislice.spec_lz, simulation.num_slices
#                 )
#             )
#             # Run the simulation
#             output_multislice = multem.simulate(
#                 system_conf, input_multislice, spec_slices
#             )

#         # Get the ideal image data
#         # Multem outputs data in column major format. In C++ and Python we
#         # generally deal with data in row major format so we must do a
#         # transpose here.
#         ideal_image = numpy.array(output_multislice.data[0].m2psi_tot)
#         if len(ideal_image) == 0:
#             ideal_image = numpy.abs(output_multislice.data[0].psi_coh).T ** 2

#         # Remove margin
#         margin = simulation.margin
#         j0 = margin
#         i0 = margin
#         j1 = ideal_image.shape[0] - margin
#         i1 = ideal_image.shape[1] - margin
#         assert margin >= 0
#         assert i1 > i0
#         assert j1 > j0
#         ideal_image = ideal_image[j0:j1, i0:i1]
#         logger.info(
#             "Ideal image min/max: %f/%f"
#             % (numpy.min(ideal_image), numpy.max(ideal_image))
#         )

#         # Compute the image scaled with Poisson noise
#         if simulation.electrons_per_pixel is not None:
#             image = (
#                 numpy.random.poisson(simulation.electrons_per_pixel * ideal_image),
#             )
#         else:
#             image = ideal_image
#         return (i, angle, image)

#     def slice_atom_data(self, atom_data, length_z, num_slices):
#         """
#         Slice the atoms into a number of subslices

#         Args:
#             atoms_data (list): The atom data
#             length_z (float): The size of the sample in Z
#             num_slices (int): The number of slices to use

#         Yields:
#             tuple: (z0, lz, atoms)

#         """

#         # Check the input
#         assert length_z > 0, length_z
#         assert num_slices > 0, num_slices

#         # The slice thickness
#         spec_lz = length_z / num_slices

#         # Get the atom z
#         atom_z = atom_data["z"]
#         min_z = numpy.min(atom_z)
#         max_z = numpy.max(atom_z)
#         assert min_z >= 0
#         assert max_z <= length_z

#         # Loop through the slices
#         for i in range(num_slices):
#             z0 = i * spec_lz
#             z1 = (i + 1) * spec_lz
#             if i == num_slices - 1:
#                 z1 = max(z1, max_z + 1)
#             selection = (atom_z >= z0) & (atom_z < z1)
#             if numpy.count_nonzero(selection) > 0:
#                 yield (
#                     z0,
#                     z1 - z0,
#                     multem.AtomList(
#                         elfantasma.sample.extract_spec_atoms(atom_data[selection])
#                     ),
#                 )

#     def atom_data(self, sample, x0, x1, y0, y1, offset):
#         """
#         Get a subset of atoms within the field of view

#         Args:
#             sample (object): The sample object
#             x0 (float): The lowest X
#             x1 (float): The highest X
#             y0 (float): The lowest Y
#             y1 (float): The highest Y
#             offset (float): The offset

#         Returns:
#             list: The spec atoms list

#         """

#         # Select the atom data
#         atom_data = sample.get_atoms_in_range([(x0, y0), (x1, y1)])

#         # Translate for the simulation
#         elfantasma.sample.translate(atom_data, (offset - x0, offset - y0, 0))

#         # Print some info
#         logger.info("Whole sample:")
#         logger.info(sample.info())
#         logger.info("Selection:")
#         logger.info("    # atoms: %d" % len(atom_data))
#         logger.info("    Min box x: %.2f" % x0)
#         logger.info("    Max box x: %.2f" % x1)
#         logger.info("    Min box y: %.2f" % y0)
#         logger.info("    Max box y: %.2f" % y1)
#         logger.info("    Min sample x: %.2f" % atom_data["x"].min())
#         logger.info("    Max sample x: %.2f" % atom_data["x"].max())
#         logger.info("    Min sample y: %.2f" % atom_data["y"].min())
#         logger.info("    Max sample y: %.2f" % atom_data["y"].max())
#         logger.info("    Min sample z: %.2f" % atom_data["z"].min())
#         logger.info("    Max sample z: %.2f" % atom_data["z"].max())

#         # Return the atom data
#         return atom_data


# class Simulation(object):
#     """
#     An object to wrap the simulation

#     """

#     def __init__(
#         self,
#         microscope=None,
#         sample=None,
#         scan=None,
#         electrons_per_pixel=None,
#         simulation_type=None,
#         margin=None,
#         freeze=None,
#         device=None,
#         slice_thickness=None,
#         cluster=None,
#         num_slices=1,
#     ):
#         """
#         Initialise the simulation

#         Args:
#             microscope (object); The microscope object
#             sample (object): The sample object
#             scan (object): The scan object

#         """

#         self.microscope = microscope
#         self.sample = sample
#         self.scan = scan
#         self.electrons_per_pixel = electrons_per_pixel
#         self.simulation_type = simulation_type
#         self.margin = margin
#         self.freeze = freeze
#         self.device = device
#         self.slice_thickness = slice_thickness
#         self.cluster = cluster
#         self.num_slices = num_slices

#     @property
#     def shape(self):
#         """
#         Return
#             tuple: The simulation data shape

#         """
#         nx = self.microscope.detector.nx
#         ny = self.microscope.detector.ny
#         nz = len(self.scan)
#         return (nz, ny, nx)

#     def run(self, writer):
#         """
#         Run the simulation

#         Args:
#             writer (object): Write each image to disk

#         """

#         # Check the shape of the writer
#         assert writer.shape == self.shape

#         # The single image simulator
#         single_image_simulator = SingleImageSimulation()

#         # If we are executing in a single process just do a for loop
#         if self.cluster["method"] is None:
#             for i, angle in enumerate(self.scan.angles):
#                 logger.info(
#                     f"    Running job: {i+1}/{self.shape[0]} for {angle} degrees"
#                 )
#                 _, angle, image = single_image_simulator(self, i)
#                 writer.data[i, :, :] = image
#                 writer.angle[i] = angle
#         else:

#             # Set the maximum number of workers
#             self.cluster["max_workers"] = min(
#                 self.cluster["max_workers"], self.shape[0]
#             )
#             logger.info("Initialising %d worker threads" % self.cluster["max_workers"])

#             # Get the futures executor
#             with elfantasma.futures.factory(**self.cluster) as executor:

#                 # Copy the data to each worker
#                 logger.info("Copying data to workers...")
#                 remote_self = executor.scatter(self, broadcast=True)

#                 # Submit all jobs
#                 logger.info("Running simulation...")
#                 futures = []
#                 for i, angle in enumerate(self.scan.angles):
#                     logger.info(
#                         f"    Submitting job: {i+1}/{self.shape[0]} for {angle} degrees"
#                     )
#                     futures.append(
#                         executor.submit(single_image_simulator, remote_self, i)
#                     )

#                 # Wait for results
#                 for j, future in enumerate(elfantasma.futures.as_completed(futures)):

#                     # Get the result
#                     i, angle, image = future.result()

#                     # Set the output in the writer
#                     writer.data[i, :, :] = image
#                     writer.angle[i] = angle

#                     # Write some info
#                     vmin = numpy.min(image)
#                     vmax = numpy.max(image)
#                     logger.info(
#                         "    Processed job: %d (%d/%d); image min/max: %.2f/%.2f"
#                         % (i + 1, j + 1, self.shape[0], vmin, vmax)
#                     )

#     def as_pickle(self, filename):
#         """
#         Write the simulated data to a python pickle file

#         Args:
#             filename (str): The output filename

#         """

#         # Make directory if it doesn't exist
#         os.makedirs(os.path.dirname(filename), exist_ok=True)

#         # Write the simulation
#         with open(filename, "wb") as outfile:
#             pickle.dump(self, outfile)


# def new(
#     microscope=None, sample=None, scan=None, device="gpu", simulation=None, cluster=None
# ):
#     """
#     Create the simulation

#     Args:
#         microscope (object); The microscope object
#         sample (object): The sample object
#         scan (object): The scan object
#         device (str): The device to use
#         simulation (object): The simulation parameters
#         cluster (object): The cluster parameters

#     Returns:
#         object: The simulation object

#     """
#     # Set the electrons per pixel
#     if microscope.beam.flux is not None:
#         beam_size = microscope.detector.nx * microscope.detector.ny
#         electrons_per_pixel = microscope.beam.flux * scan.exposure_time / beam_size
#     else:
#         electrons_per_pixel = None

#     # Set the simulation margin
#     simulation_type = simulation["type"]
#     margin = simulation["margin"]
#     freeze = simulation["freeze"]
#     slice_thickness = simulation["slice_thickness"]
#     num_slices = simulation["num_slices"]

#     # Create the simulation
#     return Simulation(
#         microscope=microscope,
#         sample=sample,
#         scan=scan,
#         electrons_per_pixel=electrons_per_pixel,
#         simulation_type=simulation_type,
#         margin=margin,
#         freeze=freeze,
#         device=device,
#         slice_thickness=slice_thickness,
#         cluster=cluster,
#         num_slices=num_slices,
#     )
