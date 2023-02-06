#
# parakeet.analyse.extract.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy as np
import mrcfile
import random
import h5py
import scipy.ndimage
import scipy.spatial.transform
import parakeet.sample
from typing import Any
from functools import singledispatch
from math import sqrt, ceil


__all__ = ["extract"]


# Set the random seed
random.seed(0)


@singledispatch
def extract(
    config_file,
    sample_file: str,
    rec_file: str,
    particles_file: str,
    particle_size: int,
):
    """
    Perform sub tomogram extraction

    Args:
        config_file: The input config filename
        sample_file: The sample filename
        rec_file: The reconstruction filename
        particles_file: The file to extract the particles to
        particle_size: The particle size (px)

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Load the sample
    sample = parakeet.sample.load(sample_file)

    # Do the sub tomogram averaging
    _extract_Config(config, sample, rec_file, particles_file, particle_size)


@extract.register(parakeet.config.Config)
def _extract_Config(
    config: parakeet.config.Config,
    sample: parakeet.sample.Sample,
    rec_file: str,
    extract_file: str,
    particle_size: int = 0,
):
    """
    Extract particles for post-processing

    """

    def rotate_array(data, rotation, offset):
        # Create the pixel indices
        az = np.arange(data.shape[0])
        ay = np.arange(data.shape[1])
        ax = np.arange(data.shape[2])
        x, y, z = np.meshgrid(az, ay, ax, indexing="ij")

        # Create a stack of coordinates
        xyz = np.vstack(
            [
                x.reshape(-1) - offset[0],
                y.reshape(-1) - offset[1],
                z.reshape(-1) - offset[2],
            ]
        ).T

        # create transformation matrix
        r = scipy.spatial.transform.Rotation.from_rotvec(rotation)

        # apply transformation
        transformed_xyz = r.apply(xyz)

        # extract coordinates
        x = transformed_xyz[:, 0] + offset[0]
        y = transformed_xyz[:, 1] + offset[1]
        z = transformed_xyz[:, 2] + offset[2]

        # Reshape
        x = x.reshape(data.shape)
        y = y.reshape(data.shape)
        z = z.reshape(data.shape)

        # sample
        result = scipy.ndimage.map_coordinates(data, [x, y, z], order=1)
        return result

    # scan = config.scan.dict()

    # Get the sample centre
    centre = np.array(sample.centre)

    # Read the reconstruction file
    tomo_file = mrcfile.mmap(rec_file)
    tomogram = tomo_file.data

    # Get the size of the volume
    voxel_size = np.array(
        (
            tomo_file.voxel_size["x"],
            tomo_file.voxel_size["y"],
            tomo_file.voxel_size["z"],
        )
    )
    size = np.array(tomogram.shape)[[2, 0, 1]] * voxel_size

    # Loop through the
    assert sample.number_of_molecules == 1
    for name, (atoms, positions, orientations) in sample.iter_molecules():
        # Compute the box size based on the size of the particle so that any
        # orientation should fit within the box
        xmin = atoms.data["x"].min()
        xmax = atoms.data["x"].max()
        ymin = atoms.data["y"].min()
        ymax = atoms.data["y"].max()
        zmin = atoms.data["z"].min()
        zmax = atoms.data["z"].max()
        xc = (xmax + xmin) / 2.0
        yc = (ymax + ymin) / 2.0
        zc = (zmax + zmin) / 2.0

        if particle_size == 0:
            half_length = (
                int(ceil(sqrt((xmin - xc) ** 2 + (ymin - yc) ** 2 + (zmin - zc) ** 2)))
                + 1
            )
        else:
            half_length = particle_size // 2
        length = 2 * half_length
        assert len(positions) == len(orientations)
        num_particles = len(positions)
        print(
            "Averaging %d %s particles with box size %d" % (num_particles, name, length)
        )

        # Create the average array
        extract_map: Any = []
        particle_instance = np.zeros(shape=(length, length, length), dtype="float32")
        num = 0

        # Sort the positions and orientations by y
        positions, orientations = zip(
            *sorted(zip(positions, orientations), key=lambda x: x[0][1])
        )

        # Loop through all the particles
        for i, (position, orientation) in enumerate(zip(positions, orientations)):
            # Compute p within the volume
            # start_position = np.array([0, scan["start_pos"], 0])
            p = position - (centre - size / 2.0)  # - start_position
            p[2] = size[2] - p[2]
            print(
                "Particle %d: position = %s, orientation = %s"
                % (
                    i,
                    "[ %.1f, %.1f, %.1f ]" % tuple(p),
                    "[ %.1f, %.1f, %.1f ]" % tuple(orientation),
                )
            )

            # Set the region to extract
            x0 = np.floor(p).astype("int32") - half_length
            x1 = np.floor(p).astype("int32") + half_length
            offset = p - np.floor(p).astype("int32")

            # Get the sub tomogram
            print("Getting sub tomogram")
            sub_tomo = tomogram[x0[1] : x1[1], x0[2] : x1[2], x0[0] : x1[0]]
            if sub_tomo.shape == particle_instance.shape:
                # Set the data to transform
                data = sub_tomo

                # Reorder input vectors
                offset = np.array(data.shape)[::-1] / 2 + offset[[1, 2, 0]]
                rotation = -np.array(orientation)[[1, 2, 0]]
                rotation[1] = -rotation[1]

                # Rotate the data
                print("Rotating volume")
                data = rotate_array(data, rotation, offset)

                # Add the contribution to the average

                extract_map.append(data)
                num += 1

        # Average the sub tomograms
        print("Extracting %d particles" % num)
        extract_map = np.array(extract_map)

        # from matplotlib import pylab
        # pylab.imshow(average[half_length, :, :])
        # pylab.show()

        # Save the averaged data
        print("Saving extracted particles to %s" % extract_file)
        handle = h5py.File(extract_file, "w")
        data_handle = handle.create_dataset("data", extract_map.shape, chunks=True)
        data_handle[:] = extract_map[:]
        handle.close()
