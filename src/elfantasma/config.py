#
# elfantasma.config.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import copy
import yaml


def temp_directory():
    """
    Returns:
        str: A temp directory for working in

    """
    return "_elfantasma"


def default_config():
    """
    Return:
        dict: the default configuration

    """
    return yaml.safe_load(
        """

        output: output.h5
        device: gpu
        phantom: 4v5d

        sample:

            4v5d:
                length_x: null
                length_y: null
                length_z: null

            ribosomes_in_lamella:
                length_x: 2000
                length_y: 2000
                length_z: 500
                number_of_ribosomes: 10

            custom:
                filename: null

        beam:
            E_0: 300
            electrons_per_pixel: 200

            objective_lens:
                m: 0
                c_10: 20
                c_12: 0.0
                phi_12: 0.0
                c_21: 0.0
                phi_21: 0.0
                c_23: 0.0
                phi_23: 0.0
                c_30: 0.04
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

        detector:
            nx: 2048
            ny: 2048

        scan:
            mode: still
            axis: 1,0,0
            start_angle: 0
            stop_angle: 360
            step_angle: 10

        simulation:
            slice_thickness: 3.0

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
                elif isinstance(value, list):
                    self[key].extend(value)
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


def load_config(config=None, command_line=None):
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
    config = deepmerge(default_config(), deepmerge(config_file, command_line))

    # Return the config
    return config


def show_config(config, full=False):
    """
    Print the command line arguments

    Args:
        config (object): The configuration object

    """
    if full == False:
        config = difference(default_config(), config)
    print("Configuration:")
    print(
        "\n".join([f"    {line}" for line in yaml.dump(config, indent=4).split("\n")])
    )
