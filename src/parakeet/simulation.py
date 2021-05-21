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
import numpy

# import pandas
import scipy.stats

# import time
import warnings
import parakeet.config
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.sample
import warnings
from math import sqrt, pi, sin, floor
from collections.abc import Iterable
from scipy.spatial.transform import Rotation

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
    input_multislice.theta = 0.0
    input_multislice.phi = 0.0

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    # Condenser lens
    # source spread function
    ssf_sigma = multem.mrad_to_sigma(
        input_multislice.E_0, microscope.beam.source_spread
    )
    input_multislice.cond_lens_ssf_sigma = ssf_sigma

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
    input_multislice.obj_lens_dsf_sigma = multem.iehwgd_to_sigma(
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
                _, angle, position, image, shift = self.simulate_image(i)
                if writer:
                    writer.data[i, :, :] = image
                    writer.angle[i] = angle
                    writer.position[i] = (0, position, 0)
                    if shift:
                        writer.shift[i] = shift
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
                    i, angle, position, image = future.result()

                    # Set the output in the writer
                    if writer:
                        writer.data[i, :, :] = image
                        writer.angle[i] = angle
                        writer.position[i] = (0, position, 0)

                    # Write some info
                    vmin = numpy.min(image)
                    vmax = numpy.max(image)
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

        # Set the rotation angle
        # input_multislice.spec_rot_theta = angle
        # input_multislice.spec_rot_u0 = simulation.scan.axis

        # Create the sample extractor
        x0 = (-offset, -offset)
        x1 = (x_fov + offset, y_fov + offset)
        thickness = self.simulation["division_thickness"]
        # extractor = parakeet.sample.AtomSliceExtractor(
        #     sample=self.sample,
        #     translation=position,
        #     rotation=angle,
        #     x0=x0,
        #     x1=x1,
        #     thickness=thickness,
        # )

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

        # Either slice or don't
        # assert len(extractor) == 1

        # Set the atoms in the input after translating them for the offset
        # zslice = extractor[0]
        atoms = self.sample.get_atoms_in_fov(x0, x1)
        logger.info("Simulating with %d atoms" % atoms.data.shape[0])
        # logger.info(
        #     "    Simulating z slice %f -> %f with %d atoms"
        #     % (zslice.x_min[2], zslice.x_max[2], zslice.atoms.data.shape[0])
        # )

        # Set atom sigma
        # atoms.data["sigma"] = sigma_B

        if len(atoms.data) > 0:
            coords = atoms.data[["x", "y", "z"]].to_numpy()
            coords = (
                Rotation.from_rotvec((0, angle * pi / 180, 0)).apply(
                    coords - self.sample.centre
                )
                + self.sample.centre
            ).astype("float32")
            atoms.data["x"] = coords[:, 0]
            atoms.data["y"] = coords[:, 1]
            atoms.data["z"] = coords[:, 2]
        # atoms.data = atoms.data.append(parakeet.sample.AtomData(atomic_number=[1,1], x=[750,750], y=[750,750], z=[0,4000],sigma=[0,0],occupancy=[1,1],charge=[0,0]).data)

        origin = (0, 0)
        input_multislice.spec_atoms = atoms.translate(
            (offset - origin[0], offset - origin[1], 0)
        ).to_multem()
        logger.info("   Got spec atoms")
        # logger.info(
        #     "    Simulating z slice %f -> %f with %d atoms"
        #     % (zslice.x_min[2], zslice.x_max[2], zslice.atoms.data.shape[0])
        # )
        # input_multislice.spec_atoms = zslice.atoms.translate(
        #     (offset, offset, 0)
        # ).to_multem()

        # Get the potential and thickness
        volume_z0 = self.sample.shape_box[0][2]
        volume_z1 = self.sample.shape_box[1][2]
        zsize = int(floor(volume_z1 - volume_z0))
        slice_thickness = self.simulation["slice_thickness"]
        handle = mrcfile.new("projected_potential_%d.mrc" % index, overwrite=True)
        handle.voxel_size = (pixel_size, pixel_size, slice_thickness)
        potential = numpy.zeros(shape=(zsize, ny, nx), dtype="float32")
        # thickness = handle.create_dataset(
        #     "thickness", (0,), dtype="float32", maxshape=(None,)
        # )
        # potential = handle.create_dataset(
        #     "potential", (0, 0, 0), dtype="float32", maxshape=(None, None, None)
        # )

        def callback(z0, z1, V):
            V = numpy.array(V)
            zc = (z0 + z1) / 2.0
            index = int(floor((zc - volume_z0) / slice_thickness))
            print(
                "Calculating potential for slice: %.2f -> %.2f (index: %d)"
                % (z0, z1, index)
            )
            potential[index, :, :] = V[margin:-margin, margin:-margin]

        # Run the simulation
        output_multislice = multem.compute_projected_potential(
            system_conf, input_multislice, callback
        )

        handle.set_data(potential)

        # Compute the image scaled with Poisson noise
        return (index, angle, position, None, None)


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

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Get the rotation angle
        angle = self.scan.angles[index]
        position = self.scan.positions[index]

        # Add the beam drift
        if (
            self.microscope.beam.drift
            and not self.microscope.beam.drift["type"] is None
        ):
            if self.microscope.beam.drift["type"] == "random":
                shiftx, shifty = numpy.random.normal(
                    0, self.microscope.beam.drift["magnitude"], size=2
                )
                logger.info("Adding drift of %f, %f " % (shiftx, shifty))
            elif self.microscope.beam.drift["type"] == "sinusoidal":
                shiftx = sin(angle * pi / 180) * self.microscope.beam.drift["magnitude"]
                shifty = shiftx
                logger.info("Adding drift of %f, %f " % (shiftx, shifty))
            else:
                raise RuntimeError("Unknown drift type")
        else:
            shiftx = 0
            shifty = 0

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        origin = numpy.array(self.microscope.detector.origin)
        margin = self.simulation["margin"]
        padding = self.simulation["padding"]
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        margin_offset = margin * pixel_size
        padding_offset = padding * pixel_size
        offset = (padding + margin) * pixel_size

        # Set the rotation angle
        # input_multislice.spec_rot_theta = angle
        # input_multislice.spec_rot_u0 = simulation.scan.axis

        # Create the sample extractor
        x0 = numpy.array((-margin_offset, position - margin_offset))
        x1 = numpy.array((x_fov + margin_offset, position + y_fov + margin_offset))
        x0 += origin
        x1 += origin
        thickness = self.simulation["division_thickness"]
        # extractor = parakeet.sample.AtomSliceExtractor(
        #    sample=self.sample,
        #    translation=position,
        #    rotation=angle,
        #    x0=x0,
        #    x1=x1,
        #    thickness=thickness,
        # )

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
            sigma_B = sqrt(
                self.simulation["sensitivity_coefficient"]
                * self.microscope.beam.electrons_per_angstrom
                * (index + 1)
            )
        else:
            sigma_B = 0

        # Either slice or don't
        if True:  # len(extractor) == 1:

            # Set the atoms in the input after translating them for the offset
            # zslice = extractor[0]
            atoms = self.sample.get_atoms_in_fov(x0, x1)
            logger.info("Simulating with %d atoms" % atoms.data.shape[0])
            # logger.info(
            #     "    Simulating z slice %f -> %f with %d atoms"
            #     % (zslice.x_min[2], zslice.x_max[2], zslice.atoms.data.shape[0])
            # )

            # Set atom sigma
            atoms.data["sigma"] = sigma_B

            if len(atoms.data) > 0:
                coords = atoms.data[["x", "y", "z"]].to_numpy()
                coords = (
                    Rotation.from_rotvec((0, angle * pi / 180, 0)).apply(
                        coords - self.sample.centre
                    )
                    + self.sample.centre
                    - (shiftx, shifty + position, 0)
                ).astype("float32")
                atoms.data["x"] = coords[:, 0]
                atoms.data["y"] = coords[:, 1]
                atoms.data["z"] = coords[:, 2]
            # atoms.data = atoms.data.append(parakeet.sample.AtomData(atomic_number=[1,1], x=[750,750], y=[750,750], z=[0,4000],sigma=[0,0],occupancy=[1,1],charge=[0,0]).data)

            input_multislice.spec_atoms = atoms.translate(
                (offset - origin[0], offset - origin[1], 0)
            ).to_multem()
            logger.info("   Got spec atoms")

            if self.simulation["ice"] == True:

                # Create the masker
                masker = multem.Masker(
                    input_multislice.nx, input_multislice.ny, pixel_size
                )

                # Get the sample centre
                shape = self.sample.shape
                centre = self.sample.centre
                centre = (
                    centre[0] + offset - shiftx - origin[0],
                    centre[1] + offset - shifty - position - origin[1],
                    centre[2],
                )

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

                # Rotate
                origin = centre
                masker.set_rotation(origin, (0, angle * pi / 180.0, 0))

                # Run the simulation
                output_multislice = multem.simulate(
                    system_conf, input_multislice, masker
                )

            else:

                # Run the simulation
                logger.info("Simulating")
                output_multislice = multem.simulate(system_conf, input_multislice)

        else:
            pass
            # # Slice the specimen atoms
            # def slice_generator(extractor):

            #     # Get the data from the data buffer and return
            #     def prepare(data_buffer):

            #         # Extract the data
            #         atoms = parakeet.sample.AtomData(
            #             data=pandas.concat([d.atoms.data for d in data_buffer])
            #         )
            #         z_min = min([d.x_min[2] for d in data_buffer])
            #         z_max = max([d.x_max[2] for d in data_buffer])
            #         assert z_min < z_max

            #         # Print some info
            #         logger.info(
            #             "    Simulating z slice %f -> %f with %d atoms"
            #             % (z_min, z_max, atoms.data.shape[0])
            #         )

            #         # Cast the atoms
            #         atoms = atoms.translate((offset, offset, 0)).to_multem()

            #         # Return the Z-min, Z-max and atoms
            #         return (z_min, z_max, atoms)

            #     # Loop through the slices and gather atoms until we have more
            #     # than the maximum buffer size. There seems to be an overhead
            #     # to the simulation code so it's better to have as many atoms
            #     # as possible before calling. Doing this is much fast than
            #     # simulating with only a small number of atoms.
            #     max_buffer = 10_000_000
            #     data_buffer = []
            #     for zslice in extractor:
            #         data_buffer.append(zslice)
            #         if sum(d.atoms.data.shape[0] for d in data_buffer) > max_buffer:
            #             yield prepare(data_buffer)
            #             data_buffer = []

            #     # Simulate from the final buffer
            #     if len(data_buffer) > 0:
            #         yield prepare(data_buffer)
            #         data_buffer = []

            # Run the simulation
            # st = time.time()
            # output_multislice = multem.simulate(
            #     system_conf, input_multislice, slice_generator(extractor)
            # )
            # logger.info(
            #     "    Image %d simulated in %d seconds" % (index, time.time() - st)
            # )

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        image = numpy.array(output_multislice.data[0].psi_coh).T
        image = image[padding:-padding, padding:-padding]

        # Print some info
        psi_tot = numpy.abs(image) ** 2
        logger.info(
            "Ideal image min/max: %f/%f" % (numpy.min(psi_tot), numpy.max(psi_tot))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image, (shiftx, shifty))


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

        def compute_image(psi, microscope, simulation, x_fov, y_fov, offset, device):

            # Create the multem system configuration
            system_conf = create_system_configuration(device)

            # Create the multem input multislice object
            input_multislice = create_input_multislice(
                microscope, simulation["slice_thickness"], simulation["margin"], "HRTEM"
            )

            # Set the specimen size
            input_multislice.spec_lx = x_fov + offset * 2
            input_multislice.spec_ly = y_fov + offset * 2
            input_multislice.spec_lz = x_fov  # self.sample.containing_box[1][2]

            # Compute and apply the CTF
            ctf = numpy.array(multem.compute_ctf(system_conf, input_multislice)).T

            # Compute the B factor for radiation damage
            if simulation["radiation_damage_model"]:
                sigma_B = sqrt(
                    simulation["sensitivity_coefficient"]
                    * microscope.beam.electrons_per_angstrom
                    * (index + 1)
                )
                pixel_size = microscope.detector.pixel_size
                Y, X = numpy.mgrid[0 : ctf.shape[0], 0 : ctf.shape[1]]
                X = (X - ctf.shape[1] // 2) / (pixel_size * ctf.shape[1])
                Y = (Y - ctf.shape[0] // 2) / (pixel_size * ctf.shape[0])
                q = numpy.sqrt(X ** 2 + Y ** 2)
                b_factor_blur = numpy.exp(-2 * pi ** 2 * q ** 2 * sigma_B ** 2)
                b_factor_blur = numpy.fft.fftshift(b_factor_blur)
                ctf = ctf * b_factor_blur

            # Compute and apply the CTF
            psi = numpy.fft.ifft2(numpy.fft.fft2(psi) * ctf)
            image = numpy.abs(psi) ** 2

            return image

        # Get the rotation angle
        angle = self.exit_wave.angle[index]
        position = self.scan.positions[index]

        # Check the angle and position
        # assert abs(angle - self.exit_wave.angle[index]) < 1e7
        assert (numpy.abs(position - self.exit_wave.position[index]) < 1e7).all()

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

        # If we do CC correction then set spherical aberration and chromatic
        # aberration to zero
        shape = self.sample["shape"]
        if self.simulation["inelastic_model"] is None:

            # If no inelastic model just calculate image as normal
            image = compute_image(
                psi, microscope, self.simulation, x_fov, y_fov, offset, self.device
            )
            electron_fraction = 1.0

        elif self.simulation["inelastic_model"] == "zero_loss":

            # Compute the image
            image = compute_image(
                psi, microscope, self.simulation, x_fov, y_fov, offset, self.device
            )

            # Calculate the fraction of electrons in the zero loss peak
            electron_fraction = parakeet.inelastic.zero_loss_fraction(shape, angle)

            # Scale the image by the fraction of electrons
            image *= electron_fraction

        elif self.simulation["inelastic_model"] == "mp_loss":

            # Set the filter width
            filter_width = self.simulation["mp_loss_width"]

            # Compute the energy and spread of the plasmon peak
            peak, sigma = parakeet.inelastic.most_probable_loss(
                microscope.beam.energy, shape, angle
            )
            if filter_width is not None:
                normal = scipy.stats.norm(loc=0, scale=sigma)
                inelastic_fraction = normal.cdf(filter_width) - normal.cdf(
                    -filter_width
                )
                sigma = scipy.stats.truncnorm.std(
                    -filter_width / 2, filter_width / 2, loc=0, scale=sigma
                )
            else:
                inelastic_fraction = 1.0
            peak /= 1000.0
            peak = min(peak, microscope.beam.energy * 0.1)
            spread = sigma / (microscope.beam.energy * 1000)

            # Add the energy loss
            microscope.beam.energy -= peak

            # Compute the energy spread of the plasmon peak
            microscope.beam.energy_spread += spread
            print("Energy: %f keV" % microscope.beam.energy)
            print("Energy spread: %f ppm" % microscope.beam.energy_spread)
            print("Inelastic fraction: % f" % inelastic_fraction)

            # Calculate the image
            image = compute_image(
                psi, microscope, self.simulation, x_fov, y_fov, offset, self.device
            )

            # Compute the fraction of electrons in the plasmon peak
            electron_fraction = parakeet.inelastic.mp_loss_fraction(shape, angle)
            electron_fraction *= inelastic_fraction

            # Scale the image by the fraction of electrons
            image *= electron_fraction

        elif self.simulation["inelastic_model"] == "unfiltered":

            # Compute the energy and spread of the plasmon peak
            peak, sigma = parakeet.inelastic.most_probable_loss(
                microscope.beam.energy, shape, angle
            )
            peak /= 1000.0
            peak = min(peak, microscope.beam.energy * 0.1)
            spread = sigma / (microscope.beam.energy * 1000)

            # Compute the zero loss image
            image1 = compute_image(
                psi, microscope, self.simulation, x_fov, y_fov, offset, self.device
            )

            # Add the energy loss
            microscope.beam.energy -= peak

            # Compute the energy spread of the plasmon peak
            microscope.beam.energy_spread += spread
            print("Energy: %f keV" % microscope.beam.energy)
            print("Energy spread: %f ppm" % microscope.beam.energy_spread)

            # Compute the MPL image
            image2 = compute_image(
                psi, microscope, self.simulation, x_fov, y_fov, offset, self.device
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
            spread = sigma / (microscope.beam.energy * 1000)

            # Compute the zero loss image
            image1 = compute_image(
                psi, microscope, self.simulation, x_fov, y_fov, offset, self.device
            )

            # Add the energy loss
            microscope.beam.energy -= peak

            # Compute the energy spread of the plasmon peak
            microscope.beam.energy_spread += spread
            print("Energy: %f keV" % microscope.beam.energy)
            print("Energy spread: %f ppm" % microscope.beam.energy_spread)

            # Compute the MPL image
            image2 = compute_image(
                psi, microscope, self.simulation, x_fov, y_fov, offset, self.device
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
            % (numpy.min(image), numpy.mean(image), numpy.max(image))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image, None)


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
        angle = self.optics.angle[index]
        position = self.scan.positions[index]

        # Check the angle and position
        # assert abs(angle - self.optics.angle[index]) < 1e7
        assert (numpy.abs(position - self.optics.position[index]) < 1e7).all()

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Compute the number of counts per pixel
        electrons_per_pixel = (
            self.microscope.beam.electrons_per_angstrom
            * self.microscope.detector.pixel_size ** 2
        )

        # Compute the electrons per pixel second
        electrons_per_second = electrons_per_pixel / self.scan.exposure_time
        energy = self.microscope.beam.energy

        # Get the image
        image = self.optics.data[index]

        # Apply the dqe in Fourier space
        if self.microscope.detector.dqe:
            logger.info("Applying DQE")
            dqe = parakeet.dqe.DQETable().dqe_fs(
                energy, electrons_per_second, image.shape
            )
            dqe = numpy.fft.fftshift(dqe)
            fft_image = numpy.fft.fft2(image)
            fft_image *= dqe
            image = numpy.real(numpy.fft.ifft2(fft_image))

        # Normalize so that the average pixel value is 1.0
        image = numpy.clip(image, 0, None)
        # image = image / numpy.mean(image) # Dont do this

        # Add Poisson noise
        # numpy.random.seed(index)
        image = numpy.random.poisson(image * electrons_per_pixel).astype("float64")

        # Print some info
        logger.info(
            "    Image min/max/mean: %g/%g/%.2g"
            % (numpy.min(image), numpy.max(image), numpy.mean(image))
        )

        # Normalize image again
        # if electrons_per_pixel > 0:
        #     image = image / electrons_per_pixel

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image.astype("float32"), None)


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
        image = numpy.array(multem.compute_ctf(system_conf, input_multislice)).T
        image = numpy.fft.fftshift(image)

        # Compute the image scaled with Poisson noise
        return (index, 0, 0, image, None)


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

        x0 = (-offset, -offset)
        x1 = (x_fov + offset, y_fov + offset)

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
        input_multislice.spec_lz = numpy.max(self.atoms.data["z"])

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
        image = numpy.array(output_multislice.data[0].psi_coh).T

        # Print some info
        psi_tot = numpy.abs(image) ** 2
        logger.info(
            "Ideal image min/max: %f/%f" % (numpy.min(psi_tot), numpy.max(psi_tot))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image, None)


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
