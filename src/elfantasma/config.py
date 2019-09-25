#
# elfantasma.config.py
#
# Copyright (C) 2019 Diamond Light Source
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import copy
import yaml


def default_config():
    """
    Return:
        dict: the default configuration

    """
    return yaml.safe_load(
        """

        output: null
        device: null
        phantom: null

        beam:
            E_0: 300
            electrons_per_pixel: 200

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

        mp_method: multiprocessing
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
