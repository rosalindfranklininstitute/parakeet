#
# elfantasma.simulation.py
#
# Copyright (C) 2019 Diamond Light Source
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import warnings
import numpy
import multem
import elfantasma.futures


class Simulation(object):
    """
    An object to wrap the simulation

    """

    def __init__(
        self,
        system_conf,
        input_multislice,
        axis=None,
        angles=None,
        electrons_per_pixel=None,
        multiprocessing=None,
    ):
        """
        Initialise the simulation

        Args:
            system_conf (object): The system configuration object
            input_multislice (object): The input object

        """
        self.system_conf = system_conf
        self.input_multislice = input_multislice
        self.axis = axis
        self.angles = angles
        self.electrons_per_pixel = electrons_per_pixel
        self.multiprocessing = multiprocessing

    def run(self, writer):
        """
        Run the simulation

        """

        # Get the futures executor
        executor = elfantasma.futures.factory(**self.multiprocessing)

        # Get the expected dimensions
        nx = self.input_multislice.nx
        ny = self.input_multislice.nx
        nz = len(self.angles)

        # Reshape the writer
        writer.shape = (nz, ny, nx)

        # Submit all jobs
        print("Running simulation...")
        results = [
            executor.submit(self.simulate_frame, i) for i, a in enumerate(self.angles)
        ]

        # Wait for results
        for i, result in enumerate(results):

            # Get the result
            angle, image = result.result()

            # Set the output in the writer
            print(f"    angle = {angle}")
            writer.data[i, :, :] = image
            writer.angle[i] = angle

    def simulate_frame(self, i):
        """
        Simulate a single frame

        Args:
            i (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        # Get the rotation angle
        angle = self.angles[i]

        # Set the rotation angle
        self.input_multislice.spec_rot_theta = angle
        self.input_multislice.spec_rot_u0 = [1, 0, 0]

        # Run the simulation
        output_multislice = multem.simulate(self.system_conf, self.input_multislice)

        # Get the ideal image data
        ideal_image = numpy.array(output_multislice.data[0].m2psi_tot)

        # Compute the image scaled with Poisson noise
        return angle, numpy.random.poisson(self.electrons_per_pixel * ideal_image)

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
        with open(filename, "wb") as outfile:
            pickle.dump(self.asdict(), outfile, protocol=2)


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


def create_simulation(
    sample,
    scan,
    device="gpu",
    beam=None,
    detector=None,
    simulation=None,
    multiprocessing=None,
):
    """
    Create the simulation

    Args:
        sample (object): The sample object
        scan (object): The scan object
        device (str): The device to use

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
    input_multislice.pn_seed = 300183

    # Set the atoms of the sample
    input_multislice.spec_atoms = sample.atom_data
    input_multislice.spec_lx = sample.length_x
    input_multislice.spec_ly = sample.length_y
    input_multislice.spec_lz = sample.length_z
    input_multislice.spec_dz = simulation["slice_thickness"]

    # Set the amorphous layers
    # input_multislice.spec_amorp = [(0, 0, 2.0)]

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"

    # x-y sampling
    input_multislice.nx = detector["nx"]
    input_multislice.ny = detector["ny"]
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
    input_multislice.obj_lens_m = 0
    input_multislice.obj_lens_c_10 = 20
    input_multislice.obj_lens_c_30 = 0.04
    input_multislice.obj_lens_c_50 = 0.00
    input_multislice.obj_lens_c_12 = 0.0
    input_multislice.obj_lens_phi_12 = 0.0
    input_multislice.obj_lens_c_23 = 0.0
    input_multislice.obj_lens_phi_23 = 0.0
    input_multislice.obj_lens_inner_aper_ang = 0.0
    input_multislice.obj_lens_outer_aper_ang = 0.0

    # defocus spread function
    dsf_sigma = multem.iehwgd_to_sigma(32)
    input_multislice.obj_lens_dsf_sigma = dsf_sigma
    input_multislice.obj_lens_dsf_npoints = 5

    # zero defocus reference
    input_multislice.obj_lens_zero_defocus_type = "First"
    input_multislice.obj_lens_zero_defocus_plane = 0

    # Set the electrons per pixel
    electrons_per_pixel = beam["electrons_per_pixel"]

    # Create the simulation
    return Simulation(
        system_conf,
        input_multislice,
        scan.axis,
        scan.angles,
        electrons_per_pixel,
        multiprocessing,
    )
