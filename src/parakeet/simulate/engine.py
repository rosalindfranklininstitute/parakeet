#
# parakeet.simulate.engine.py
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
from math import sqrt, floor
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


class SimulationEngine(object):
    """
    A class to encapsulate the multem stuff

    """

    def __init__(
        self,
        device,
        gpu_id,
        microscope,
        slice_thickness,
        margin,
        simulation_type,
        centre=None,
    ):
        """
        Initialise the simulation engine

        """

        # Save the margin
        self.margin = margin

        # Setup the system configuration
        self.system_conf = self._create_system_configuration(device, gpu_id)

        # Setup the input multislice
        if simulation_type in ["CBED"]:
            self.input = self._create_input_multislice_diffraction(
                microscope, slice_thickness, margin, simulation_type, centre
            )
        else:
            self.input = self._create_input_multislice(
                microscope, slice_thickness, margin, simulation_type, centre
            )

    def _create_system_configuration(self, device, gpu_id):
        """
        Create an appropriate system configuration

        Args:
            device (str): The device to use
            gpu_id (int): The gpu id

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

        # Set the GPU ID
        if gpu_id is not None:
            system_conf.gpu_device = gpu_id

        # Print some output
        logger.info("Simulating using %s" % system_conf.device)

        # Return the system configuration
        return system_conf

    def _create_input_multislice(
        self, microscope, slice_thickness, margin, simulation_type, centre=None
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
        input_multislice.illumination_model = "Partial_Coherent_Higher_Order"
        input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

        # Condenser lens
        # source spread function
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

        # Set the incident wave
        if microscope.beam.incident_wave is not None:
            assert microscope.beam.incident_wave.shape[0] == input_multislice.ny
            assert microscope.beam.incident_wave.shape[1] == input_multislice.nx
            input_multislice.iw_type = "User_Define_Wave"
            input_multislice.iw_psi = microscope.beam.incident_wave.T.flatten()

        # Return the input multislice object
        return input_multislice

    def _create_input_multislice_diffraction(
        self, microscope, slice_thickness, margin, simulation_type, centre=None
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

        # Set the incident wave
        if microscope.beam.incident_wave is not None:
            assert microscope.beam.incident_wave.shape[0] == input_multislice.ny
            assert microscope.beam.incident_wave.shape[1] == input_multislice.nx
            input_multislice.iw_type = "User_Define_Wave"
            input_multislice.iw_psi = microscope.beam.incident_wave.T.flatten()

        # Return the input multislice object
        return input_multislice

    def ctf(self):
        """
        Simulate the CTF

        """
        return np.array(multem.compute_ctf(self.system_conf, self.input)).T

    def potential(self, out, volume_z0, masker=None):
        """
        Simulate the potential

        """
        # Set the incident wave
        self.input.iw_x = [self.input.spec_lx / 2]
        self.input.iw_y = [self.input.spec_ly / 2]

        margin = self.margin
        slice_thickness = self.input.spec_dz

        def callback(z0, z1, V):
            V = np.array(V)
            zc = (z0 + z1) / 2.0
            index = int(floor((zc - volume_z0) / slice_thickness))
            print(
                "Calculating potential for slice: %.2f -> %.2f (index: %d)"
                % (z0, z1, index)
            )
            if index < out.data.shape[0]:
                x0 = margin
                y0 = margin
                x1 = V.shape[1] - margin
                y1 = V.shape[0] - margin
                out.data[index, :, :] = V[y0:y1, x0:x1].T

        # Run the simulation
        if masker is not None:
            multem.compute_projected_potential(
                self.system_conf, self.input, masker, callback
            )
        else:
            multem.compute_projected_potential(self.system_conf, self.input, callback)

    def image(self, masker=None):
        """
        Simulate the image

        """
        # Set the incident wave
        self.input.iw_x = [self.input.spec_lx / 2]
        self.input.iw_y = [self.input.spec_ly / 2]

        # Run the simulation
        if masker is not None:
            output_multislice = multem.simulate(self.system_conf, self.input, masker)
        else:
            output_multislice = multem.simulate(self.system_conf, self.input)

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        return np.array(output_multislice.data[0].psi_coh).T

    def diffraction_image(self, masker=None):
        """
        Simulate the image

        """
        # Set the incident wave
        # For some reason need this to work with CBED
        self.input.iw_x = [self.input.spec_lx / 2]
        self.input.iw_y = [self.input.spec_ly / 2]

        # Run the simulation
        if masker is not None:
            output_multislice = multem.simulate(self.system_conf, self.input, masker)
        else:
            output_multislice = multem.simulate(self.system_conf, self.input)

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        return np.array(output_multislice.data[0].m2psi_tot).T

    def masker(
        self,
        index,
        pixel_size,
        origin,
        offset,
        orientation,
        shift,
        sample,
        scan,
        simulation,
    ):
        """
        Get the masker object for the ice specification

        """

        # Create the masker
        masker = multem.Masker(self.input.nx, self.input.ny, pixel_size)

        # Set the ice parameters
        ice_parameters = multem.IceParameters()
        ice_parameters.m1 = simulation["ice_parameters"]["m1"]
        ice_parameters.m2 = simulation["ice_parameters"]["m2"]
        ice_parameters.s1 = simulation["ice_parameters"]["s1"]
        ice_parameters.s2 = simulation["ice_parameters"]["s2"]
        ice_parameters.a1 = simulation["ice_parameters"]["a1"]
        ice_parameters.a2 = simulation["ice_parameters"]["a2"]
        ice_parameters.density = simulation["ice_parameters"]["density"]
        masker.set_ice_parameters(ice_parameters)

        # Get the sample centre
        shape = sample.shape
        centre = np.array(sample.centre)
        detector_origin = np.array([origin[0], origin[1], 0])
        centre = centre + offset - detector_origin - shift

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
        if scan.is_uniform_angular_scan:
            masker.set_rotation(centre, (0, 0, 0))
        else:
            masker.set_rotation(centre, orientation)

        # Get the masker
        return masker
