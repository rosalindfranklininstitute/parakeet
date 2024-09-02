#
# parakeet.metadata.export.py
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
import pandas as pd
import parakeet.config
import parakeet.dqe
import parakeet.microscope
import parakeet.io
import os
import starfile
from functools import singledispatch
from math import pi
from parakeet.sample import Sample


__all__ = ["export"]


# Get the logger
logger = logging.getLogger(__name__)


class RelionMetadataExporter(object):
    """
    A class to help export metadata for use with relion

    """

    def __init__(
        self,
        config: parakeet.config.Config,
        sample: Sample,
        image: parakeet.io.Reader,
        directory: str,
    ):
        """
        Initialise the exporter

        """

        # Save the config and sample
        self.config = config
        self.sample = sample
        self.image = image

        # Set the relion metadata output
        self.directory = os.path.join(directory, "relion")

        # Create the directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Setup some filenames
        self.input_filename = os.path.join(self.directory, "relion_input.star")
        self.mtf_filename = os.path.join(
            self.directory, "mtf_%dkV.star" % int(config.microscope.beam.energy)
        )
        self.corrected_micrographs_filename = os.path.join(
            self.directory, "corrected_micrographs.star"
        )
        self.single_particle_scan_filename = os.path.join(
            self.directory, "particle.star"
        )

    def write_input_file(self):
        """
        Write out a star file with relion input

        """
        microscope = self.config.microscope

        # Create the dictionary
        data = {
            "relion_input": pd.DataFrame.from_dict(
                {
                    "rlnPixelSize": [microscope.detector.pixel_size],
                    "rlnVoltage": [microscope.beam.energy],
                    "rlnSphericalAberration": [microscope.lens.c_30],
                    "rlnAmplitudeContrast": [0.1],
                    "rlnBeamTiltX": [microscope.beam.phi * 1e3 * pi / 180],
                    "rlnBeamTiltY": [microscope.beam.theta * 1e3 * pi / 180],
                }
            )
        }

        # Write the file
        starfile.write(data, self.input_filename, overwrite=True)

    def write_mtf_file(self):
        """
        Write out a star file with mtf

        """
        microscope = self.config.microscope

        # Get the MTF
        if microscope.detector.dqe:
            electrons_per_second = 1.0  # FIXME Use exposure time
            dqe = parakeet.dqe.DQETable()
            q = dqe.spatial_freq
            mtf = dqe.dqe_table(microscope.beam.energy, electrons_per_second)
        else:
            N = 200
            q = np.arange(N) * 0.5 / (N - 1)
            mtf = np.ones(q.shape)

        # Name of the dataset
        name = os.path.basename(os.path.splitext(self.mtf_filename)[0])

        # Create the dictionary
        data = {
            name: pd.DataFrame.from_dict(
                {"rlnResolutionInversePixel": q, "rlnMtfValue": mtf}
            )
        }

        # Write the file
        starfile.write(data, self.mtf_filename, overwrite=True)

    def write_corrected_micrographs_file(self):
        """
        Write out a star file with corrected micrographs data

        """
        microscope = self.config.microscope

        # Create the dictionary
        data = {
            "optics": pd.DataFrame.from_dict(
                {
                    "rlnOpticsGroupName": ["opticsGroup1"],
                    "rlnOpticsGroup": [1],
                    "rlnMtfFileName": [self.mtf_filename],
                    "rlnMicrographOriginalPixelSize": [microscope.detector.pixel_size],
                    "rlnVoltage": [microscope.beam.energy],
                    "rlnSphericalAberration": [microscope.lens.c_30],
                    "rlnAmplitudeContrast": [0.1],
                    "rlnMicrographPixelSize": [microscope.detector.pixel_size],
                }
            ),
            "micrographs": pd.DataFrame.from_dict(
                {
                    "rlnCtfPowerSpectrum": [""],
                    "rlnMicrographName": [""],
                    "rlnMicrographMetadata": [""],
                    "rlnOpticsGroup": [1],
                    "rlnAccumMotionTotal": [0],
                    "rlnAccumMotionEarly": [0],
                    "rlnAccumMotionLate": [0],
                }
            ),
        }

        # Write the file
        starfile.write(data, self.corrected_micrographs_filename, overwrite=True)

    def write_manual_pick_files(self):
        """
        Write out a star file with picking metadata

        """

        for i in range(1, self.config.scan.num_images + 1):
            # Create the dictionary
            data = {
                "picking": pd.DataFrame.from_dict(
                    {
                        "rlnCoordinateX": 0,
                        "rlnCoordinateY": 0,
                        "rlnClassNumber": 1,
                        "rlnAnglePsi": -999,
                        "rlnAutopickFigureOfMerit": -999,
                    }
                ),
            }

            # Write the file
            starfile.write(data, self.manual_pick_filename % i, overwrite=True)

    def write_single_particle_scan_files(self):
        """
        Write out a star file with single particle simulation metadata

        """
        scan = self.image.header.scan
        relion_eulers = scan.euler_angles
        relion_shifts = -scan.shift
        data = {
            "rlnAngleRot": relion_eulers[:, 0],
            "rlnAngleTilt": relion_eulers[:, 1],
            "rlnAnglePsi": relion_eulers[:, 2],
            "rlnOriginX": relion_shifts[:, 0],
            "rlnOriginY": relion_shifts[:, 1],
            "rlnOriginZ": relion_shifts[:, 2],
        }
        df = pd.DataFrame.from_dict(data)
        starfile.write(df, self.single_particle_scan_filename, overwrite=True)


def export_relion(
    config: parakeet.config.Config,
    sample: Sample,
    image: parakeet.io.Reader,
    directory: str,
):
    """
    Export metadata

    Args:
        config: The input config
        sample: The input sample
        image: The input image data
        directory: The output directory

    """

    # Create the relion exporter and export the data
    exporter = RelionMetadataExporter(config, sample, image, directory)
    exporter.write_input_file()
    exporter.write_mtf_file()
    exporter.write_corrected_micrographs_file()
    exporter.write_single_particle_scan_files()


@singledispatch
def export(
    config_file,
    sample_file: str,
    image_file: str,
    directory: str = ".",
    relion: bool = True,
):
    """
    Export metadata

    Args:
        config_file: The input config filename
        sample_file: The input sample filename
        image_file: The input image filename
        directory: The output directory
        relion: True/False output the relion metadata

    """
    # Load the configuration
    config = parakeet.config.load(config_file)

    # Open the sample
    sample = Sample(sample_file, mode="r")

    image = parakeet.io.open(image_file)

    # Export the metadata
    return _export_Config(config, sample, image, directory, relion)


@export.register(parakeet.config.Config)
def _export_Config(
    config: parakeet.config.Config,
    sample: Sample,
    image: parakeet.io.Reader,
    directory: str = ".",
    relion: bool = True,
):
    """
    Export metadata

    Args:
        config: The input config
        sample: The input sample
        image: The input image
        directory: The output directory
        relion: True/False output the relion metadata

    """

    # Export relion metadata
    if relion:
        export_relion(config, sample, image, directory)
