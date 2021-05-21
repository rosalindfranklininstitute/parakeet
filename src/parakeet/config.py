#
# parakeet.config.py
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
import yaml

# Get the logger
logger = logging.getLogger(__name__)


def temp_directory():
    """
    Returns:
        str: A temp directory for working in

    """
    return "_parakeet"


def default():
    """
    Return:
        dict: the default configuration

    """
    return yaml.safe_load(
        """

        sample:

            coords:
                filename: null
                recentre: False

            box: [ 4000, 4000, 4000 ]

            centre: [ 2000, 2000, 2000 ]

            shape:
                type: cube

                cube:
                    length: 4000

                cuboid:
                    length_x: 4000
                    length_y: 4000
                    length_z: 4000

                cylinder:
                    length: 10000
                    radius: 1500

                margin: [ 0, 0, 0 ]

            ice:
                generate: False
                density: 940

            molecules:

                4v5d: 0
                4v1w: 0
                6qt9: 0
                6z6u: 0

            sputter:
                element: null
                thickness: 20

        microscope:

            model: null

            beam:
                energy: 300
                energy_spread: 2.66e-6
                acceleration_voltage_spread: 0.8e-6
                electrons_per_angstrom: 30
                source_spread: 0.1
                drift:
                    type: null
                    magnitude: 0

            objective_lens:
                m: 0
                c_10: 20000
                c_12: 0.0
                phi_12: 0.0
                c_21: 0.0
                phi_21: 0.0
                c_23: 0.0
                phi_23: 0.0
                c_30: 2.7
                c_32: 0.0
                phi_32: 0.0
                c_34: 0.0
                phi_34: 0.0
                c_41: 0.0
                phi_41: 0.0
                c_43: 0.0
                phi_43: 0.0
                c_45: 0.0
                phi_45: 0.0
                c_50: 0.0
                c_52: 0.0
                phi_52: 0.0
                c_54: 0.0
                phi_54: 0.0
                c_56: 0.0
                phi_56: 0.0
                inner_aper_ang: 0.0
                outer_aper_ang: 0.0
                c_c: 2.7
                current_spread: 0.33e-6

            phase_plate: false

            detector:
                nx: 4000
                ny: 4000
                pixel_size: 1
                dqe: False
                origin: [ 0, 0 ]

        scan:
            mode: still
            axis: [0,1,0]
            start_angle: 0
            step_angle: 10
            start_pos: 0
            step_pos: auto
            num_images: 1
            exposure_time: 1

        device: gpu

        simulation:
            slice_thickness: 3.0
            margin: 100
            padding: 100
            division_thickness: 100
            ice: False
            radiation_damage_model: False
            inelastic_model: null
            mp_loss_width: null
            sensitivity_coefficient: 0.014

        cluster:
            method: null
            max_workers: 1

    """
    )


def deepmerge(a, b):
    """
    Perform a deep merge of two dictionaries

    Args:
        a (dict): The first dictionary
        b (dict): The second dictionary

    Returns:
        dict: The merged dictionary

    """

    def deepmerge_internal(self, other):
        for key, value in other.items():
            if key in self:
                if isinstance(value, dict):
                    deepmerge_internal(self[key], value)
                else:
                    self[key] = copy.deepcopy(value)
            else:
                self[key] = copy.deepcopy(value)
        return self

    return deepmerge_internal(copy.deepcopy(a), b)


def difference(master, config):
    """
    Get the differernce between the master and config dictionaries

    Get the difference between values in the master and configuration
    dictionaries. Each value is checked recursively. Since lists are appended,
    any list is set as a difference.

    Args:
        master (dict): The master dictionary
        config (dict): The configuration dictionary

    Returns:
        dict: The difference dictionary

    """

    def walk(master, config):
        result = {}
        for key, value in config.items():
            if key in master:
                if isinstance(value, dict):
                    value = walk(master[key], value)
                    if len(value) > 0:
                        result[key] = value
                elif isinstance(value, list):
                    result[key] = value
                elif master[key] != value:
                    result[key] = value
            else:
                result[key] = value
        return result

    return walk(master, config)


def load(config=None, command_line=None):
    """
    Load the configuration from the various inputs

    Args:
        config (str): The config filename
        command_line (dict): The command line arguments

    Returns:
        dict: The configuration dictionary

    """

    # Get the command line arguments
    if command_line is None:
        command_line = {}

    # If the yaml configuration is set then merge the configuration
    if config:
        with open(config) as infile:
            config_file = yaml.safe_load(infile)
    else:
        config_file = {}

    # Get the configuration
    config = deepmerge(default(), deepmerge(config_file, command_line))

    # Return the config
    return config


def show(config, full=False):
    """
    Print the command line arguments

    Args:
        config (object): The configuration object

    """
    if full == False:
        config = difference(default(), config)
    logger.info(
        "\n".join([f"{line}" for line in yaml.dump(config, indent=4).split("\n")])
    )
