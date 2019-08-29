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


class Simulation(object):
    """
    An object to wrap the simulation

    """

    def __init__(self, system_conf, input_multislice):
        """
        Initialise the simulation

        Args:
            system_conf (object): The system configuration object
            input_multislice (object): The input object

        """
        self.system_conf = system_conf
        self.input_multislice = input_multislice

    def run(self):
        """
        Run the simulation

        """
        print("Running simulation...")
        self.output_multislice = multem.simulate(
            self.system_conf, self.input_multislice
        )

    def asdict(self):
        """
        Returns:
            dict: The results as a dictionary

        """
        return {
            "input": self.input_multislice.asdict(),
            "output": self.output_multislice.asdict(),
        }


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


def create_simulation(sample, device="gpu"):
    """
    Create the simulation

    Args:
        sample (object): The sample object
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
    input_multislice.potential_slicing = "Planes"

    # Electron-Phonon interaction model
    input_multislice.pn_model = "Frozen_Phonon"
    input_multislice.pn_coh_contrib = 0
    input_multislice.pn_single_conf = False
    input_multislice.pn_nconf = 10
    input_multislice.pn_dim = 110
    input_multislice.pn_seed = 300183

    # Set the atoms of the sample
    input_multislice.spec_atoms = list(sample)
    input_multislice.spec_lx = 500
    input_multislice.spec_ly = 500
    input_multislice.spec_lz = 500
    input_multislice.spec_dz = 1
    c = 3

    # Set the amorphous layers
    input_multislice.spec_amorp = [(0, 0, 2.0)]

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"
    input_multislice.thick = numpy.arange(c, 1000, c)

    # x-y sampling
    input_multislice.nx = 1024
    input_multislice.ny = 1024
    input_multislice.bwl = False

    # Microscope parameters
    input_multislice.E_0 = 300
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

    # Return the simulation object
    return Simulation(system_conf, input_multislice)
