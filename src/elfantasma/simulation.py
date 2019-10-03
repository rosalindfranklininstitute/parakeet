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
import multem
import numpy
import os
import pickle
import warnings
import elfantasma.config
import elfantasma.futures
import elfantasma.sample


class SingleImageSimulation(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __call__(self, simulation, i):
        """
        Simulate a single frame

        Args:
            simulation (object): The simulation object
            i (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        # Get the rotation angle
        angle = simulation.angles[i]
        position = simulation.positions[i]

        # Set the rotation angle
        simulation.input_multislice.spec_rot_theta = angle
        simulation.input_multislice.spec_rot_u0 = simulation.axis

        # The field of view
        x_fov = simulation.detector["nx"] * simulation.detector["pixel_size"]
        y_fov = simulation.detector["ny"] * simulation.detector["pixel_size"]
        offset = simulation.margin * simulation.detector["pixel_size"]

        # Get the specimen atoms
        print(f"Simulating image {i}")
        spec_atoms = self.spec_atoms(
            simulation.sample, position, position + x_fov, 0, y_fov, offset
        )

        # Set the atoms of the sample
        simulation.input_multislice.spec_atoms = spec_atoms
        simulation.input_multislice.spec_lx = x_fov + offset * 2
        simulation.input_multislice.spec_ly = y_fov + offset * 2
        simulation.input_multislice.spec_lz = simulation.sample.box_size[2]

        # Run the simulation
        output_multislice = multem.simulate(
            simulation.system_conf, simulation.input_multislice
        )

        # Get the ideal image data
        ideal_image = numpy.array(output_multislice.data[0].m2psi_tot).T

        # Remove margin
        margin = simulation.margin
        ideal_image = ideal_image[margin:-margin, margin:-margin]
        print(
            "Ideal image min/max: %f/%f"
            % (numpy.min(ideal_image), numpy.max(ideal_image))
        )

        # Compute the image scaled with Poisson noise
        return (
            i,
            angle,
            numpy.random.poisson(simulation.electrons_per_pixel * ideal_image),
        )

    def spec_atoms(self, sample, x0, x1, y0, y1, offset):
        """
        Get a subset of atoms within the field of view

        Args:
            sample (object): The sample object
            x0 (float): The lowest X
            x1 (float): The highest X
            y0 (float): The lowest Y
            y1 (float): The highest Y
            offset (float): The offset

        Returns:
            list: The spec atoms list

        """

        # Select the atom data
        atom_data = sample.select_atom_data_in_roi([(x0, y0), (x1, y1)])

        # Translate for the simulation
        elfantasma.sample.translate(atom_data, (offset - x0, offset - y0, 0))

        # Print some info
        print("Whole sample:")
        print(sample.info())
        print("Selection:")
        print("    # atoms: %d" % len(atom_data))
        print("    Min box x: %.2f" % x0)
        print("    Max box x: %.2f" % x1)
        print("    Min box y: %.2f" % y0)
        print("    Max box x: %.2f" % y1)
        print("    Min sample x: %.2f" % atom_data["x"].min())
        print("    Max sample x: %.2f" % atom_data["x"].max())
        print("    Min sample y: %.2f" % atom_data["y"].min())
        print("    Max sample y: %.2f" % atom_data["y"].max())
        print("    Min sample z: %.2f" % atom_data["z"].min())
        print("    Max sample z: %.2f" % atom_data["z"].max())

        # Return the atom data
        return list(elfantasma.sample.extract_spec_atoms(atom_data))


class Simulation(object):
    """
    An object to wrap the simulation

    """

    def __init__(
        self,
        system_conf,
        input_multislice,
        sample=None,
        detector=None,
        axis=None,
        angles=None,
        positions=None,
        electrons_per_pixel=None,
        margin=None,
        cluster=None,
    ):
        """
        Initialise the simulation

        Args:
            system_conf (object): The system configuration object
            input_multislice (object): The input object

        """

        # Check the input
        assert len(angles) == len(positions)

        self.system_conf = system_conf
        self.input_multislice = input_multislice
        self.sample = sample
        self.detector = detector
        self.axis = axis
        self.angles = angles
        self.positions = positions
        self.electrons_per_pixel = electrons_per_pixel
        self.margin = margin
        self.cluster = cluster

    @property
    def shape(self):
        """
        Return the simulation data shape

        """
        nx = self.detector["nx"]
        ny = self.detector["nx"]
        nz = len(self.angles)
        return (nz, ny, nx)

    def run(self, writer):
        """
        Run the simulation

        """

        # Check the shape of the writer
        assert writer.shape == self.shape

        # The single image simulator
        single_image_simulator = SingleImageSimulation()

        # If we are executing in a single process just do a for loop
        if self.cluster["method"] is None:
            for i, angle in enumerate(self.angles):
                print(f"    Running job: {i+1}/{self.shape[0]} for {angle} degrees")
                _, angle, image = single_image_simulator(self, i)
                writer.data[i, :, :] = image
                writer.angle[i] = angle
        else:

            # Set the maximum number of workers
            self.cluster["max_workers"] = min(
                self.cluster["max_workers"], self.shape[0]
            )
            print("Initialising %d worker threads" % self.cluster["max_workers"])

            # Get the futures executor
            with elfantasma.futures.factory(**self.cluster) as executor:

                # Copy the data to each worker
                print("Copying data to workers...")
                remote_self = executor.scatter(self, broadcast=True)

                # Submit all jobs
                print("Running simulation...")
                futures = []
                for i, angle in enumerate(self.angles):
                    print(
                        f"    Submitting job: {i+1}/{self.shape[0]} for {angle} degrees"
                    )
                    futures.append(
                        executor.submit(single_image_simulator, remote_self, i)
                    )

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
                    print(
                        "    Processed job: %d (%d/%d); image min/max: %.2f/%.2f"
                        % (i + 1, j + 1, self.shape[0], vmin, vmax)
                    )

    def asdict(self):
        """
        Returns:
            dict: The results as a dictionary

        """
        return {"input": self.input_multislice.asdict(), "angles": self.angles}

    def as_pickle(self, filename):
        """
        Write the simulated data to a python pickle file

        Args:
            filename (str): The output filename

        """

        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Write the simulation
        with open(filename, "wb") as outfile:
            pickle.dump(self, outfile)


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

    # Return the system configuration
    return system_conf


def new(
    sample, scan, device="gpu", beam=None, detector=None, simulation=None, cluster=None
):
    """
    Create the simulation

    Args:
        sample (object): The sample object
        scan (object): The scan object
        device (str): The device to use
        beam (object); The beam parameters
        detector (object): The detector parameters
        simulation (object): The simulation parameters
        cluster (object): The cluster parameters

    Returns:
        object: The simulation object

    """

    # Create the system configuration
    system_conf = create_system_configuration(device)

    # Initialise the input and system configuration
    input_multislice = multem.Input()

    # Set simulation experiment
    input_multislice.simulation_type = "HRTEM"

    # Electron-Specimen interaction model
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_type = "Lobato_0_12"

    # Potential slicing
    input_multislice.potential_slicing = "dz_Proj"

    # Electron-Phonon interaction model
    input_multislice.pn_model = "Frozen_Phonon"
    input_multislice.pn_coh_contrib = 0
    input_multislice.pn_single_conf = False
    input_multislice.pn_nconf = 10
    input_multislice.pn_dim = 110
    input_multislice.pn_seed = 300_183

    # Set the slice thickness
    input_multislice.spec_dz = simulation["slice_thickness"]

    # Set the amorphous layers
    # input_multislice.spec_amorp = [(0, 0, 2.0)]

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"

    # x-y sampling
    input_multislice.nx = detector["nx"] + simulation["margin"] * 2
    input_multislice.ny = detector["ny"] + simulation["margin"] * 2
    input_multislice.bwl = False

    # Microscope parameters
    input_multislice.E_0 = beam["E_0"]
    input_multislice.theta = 0.0
    input_multislice.phi = 0.0

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    # Condenser lens
    # source spread function
    ssf_sigma = multem.mrad_to_sigma(input_multislice.E_0, 0.02)
    # input_multislice.obj_lens_ssf_sigma = ssf_sigma
    # input_multislice.obj_lens_ssf_npoints = 4

    # Objective lens
    input_multislice.obj_lens_m = beam["objective_lens"]["m"]
    input_multislice.obj_lens_c_10 = beam["objective_lens"]["c_10"]
    input_multislice.obj_lens_c_12 = beam["objective_lens"]["c_12"]
    input_multislice.obj_lens_phi_12 = beam["objective_lens"]["phi_12"]
    input_multislice.obj_lens_c_21 = beam["objective_lens"]["c_21"]
    input_multislice.obj_lens_phi_21 = beam["objective_lens"]["phi_21"]
    input_multislice.obj_lens_c_23 = beam["objective_lens"]["c_23"]
    input_multislice.obj_lens_phi_23 = beam["objective_lens"]["phi_23"]
    input_multislice.obj_lens_c_30 = beam["objective_lens"]["c_30"]
    input_multislice.obj_lens_c_32 = beam["objective_lens"]["c_32"]
    input_multislice.obj_lens_phi_32 = beam["objective_lens"]["phi_32"]
    input_multislice.obj_lens_c_34 = beam["objective_lens"]["c_34"]
    input_multislice.obj_lens_phi_34 = beam["objective_lens"]["phi_34"]
    input_multislice.obj_lens_c_41 = beam["objective_lens"]["c_41"]
    input_multislice.obj_lens_phi_41 = beam["objective_lens"]["phi_41"]
    input_multislice.obj_lens_c_43 = beam["objective_lens"]["c_43"]
    input_multislice.obj_lens_phi_43 = beam["objective_lens"]["phi_43"]
    input_multislice.obj_lens_c_45 = beam["objective_lens"]["c_45"]
    input_multislice.obj_lens_phi_45 = beam["objective_lens"]["phi_45"]
    input_multislice.obj_lens_c_50 = beam["objective_lens"]["c_50"]
    input_multislice.obj_lens_c_52 = beam["objective_lens"]["c_52"]
    input_multislice.obj_lens_phi_52 = beam["objective_lens"]["phi_52"]
    input_multislice.obj_lens_c_54 = beam["objective_lens"]["c_54"]
    input_multislice.obj_lens_phi_54 = beam["objective_lens"]["phi_54"]
    input_multislice.obj_lens_c_56 = beam["objective_lens"]["c_56"]
    input_multislice.obj_lens_phi_56 = beam["objective_lens"]["phi_56"]
    input_multislice.obj_lens_inner_aper_ang = beam["objective_lens"]["inner_aper_ang"]
    input_multislice.obj_lens_outer_aper_ang = beam["objective_lens"]["outer_aper_ang"]

    # defocus spread function
    dsf_sigma = multem.iehwgd_to_sigma(32)
    input_multislice.obj_lens_dsf_sigma = dsf_sigma
    input_multislice.obj_lens_dsf_npoints = 5

    # zero defocus reference
    input_multislice.obj_lens_zero_defocus_type = "First"
    input_multislice.obj_lens_zero_defocus_plane = 0

    # Set the electrons per pixel
    electrons_per_pixel = beam["electrons_per_pixel"]

    # Set the simulation margin
    margin = simulation["margin"]

    # Create the simulation
    return Simulation(
        system_conf,
        input_multislice,
        sample,
        detector,
        scan.axis,
        scan.angles,
        scan.positions,
        electrons_per_pixel,
        margin,
        cluster,
    )
