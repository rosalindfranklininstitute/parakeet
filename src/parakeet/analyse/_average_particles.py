#
# parakeet.analyse.average_particles.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import concurrent.futures
import numpy as np
import mrcfile
import random
import scipy.ndimage
import scipy.spatial.transform
import parakeet.sample
from functools import singledispatch
from math import sqrt, ceil


__all__ = ["average_particles", "average_all_particles"]


# Set the random seed
random.seed(0)


def _rotate_array(data, rotation, offset):
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


def _process_sub_tomo(args):
    sub_tomo, position, orientation, half_index = args

    # Set the data to transform
    data = sub_tomo
    offset = position

    # Reorder input vectors
    offset = np.array(data.shape)[::-1] / 2 + offset[[1, 2, 0]]
    rotation = -np.array(orientation)[[1, 2, 0]]
    rotation[1] = -rotation[1]

    # Rotate the data
    print("Rotating volume")
    data = _rotate_array(data, rotation, offset)
    return half_index, data


def _iterate_particles(
    indices,
    positions,
    orientations,
    centre,
    size,
    half_length,
    half_shape,
    voxel_size,
    tomogram,
):
    for j in range(len(indices)):
        for i in indices[j]:
            position = positions[i]
            orientation = orientations[i]

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
            x0 = np.floor(p / voxel_size).astype("int32") - half_length
            x1 = np.floor(p / voxel_size).astype("int32") + half_length
            offset = p - np.floor(p).astype("int32")

            # Get the sub tomogram
            print("Getting sub tomogram")
            sub_tomo = tomogram[x0[1] : x1[1], x0[2] : x1[2], x0[0] : x1[0]]
            if sub_tomo.shape == half_shape[-3:]:
                yield (sub_tomo, offset, orientation, j)


def lazy_map(executor, func, iterable):
    """
    A lazy map function for concurrent processes

    """
    max_workers = executor._max_workers

    futures = []
    for it in iterable:
        futures.append(executor.submit(func, it))
        while len(futures) >= max_workers:
            temp = []
            for f in futures:
                if f.done():
                    yield f.result()
                else:
                    temp.append(f)
            futures = temp

    for future in concurrent.futures.as_completed(futures):
        yield future.result()


@singledispatch
def average_particles(
    config_file,
    sample_file: str,
    rec_file: str,
    half1_file: str,
    half2_file: str,
    particle_size: int,
    num_particles: int,
):
    """
    Perform sub tomogram averaging

    Args:
        config_file: The input config filename
        sample_file: The sample filename
        rec_file: The reconstruction filename
        half1_file: The particle average filename for half 1
        half2_file: The particle average filename for half 2
        particle_size: The particle size (px)
        num_particles: The number of particles to average

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Load the sample
    sample = parakeet.sample.load(sample_file)

    # Do the sub tomogram averaging
    _average_particles_Config(
        config.scan,
        sample,
        rec_file,
        half1_file,
        half2_file,
        particle_size,
        num_particles,
    )


@average_particles.register(parakeet.config.Scan)
def _average_particles_Config(
    config: parakeet.config.Scan,
    sample: parakeet.sample.Sample,
    rec_filename: str,
    half_1_filename: str,
    half_2_filename: str,
    particle_size: int = 0,
    num_particles: int = 0,
):
    """
    Average particles to compute averaged reconstruction

    """

    # Get the scan dict
    # scan = config.model_dump()

    # Get the sample centre
    centre = np.array(sample.centre)

    # Read the reconstruction file
    tomo_file = mrcfile.mmap(rec_filename)
    tomogram = tomo_file.data

    # Get the size of the volume
    voxel_size = np.array(
        (
            tomo_file.voxel_size["x"],
            tomo_file.voxel_size["y"],
            tomo_file.voxel_size["z"],
        )
    )
    assert voxel_size[0] == voxel_size[1]
    assert voxel_size[0] == voxel_size[2]
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
                int(
                    ceil(
                        sqrt(
                            ((xmin - xc) / voxel_size[0]) ** 2
                            + ((ymin - yc) / voxel_size[1]) ** 2
                            + ((zmin - zc) / voxel_size[2]) ** 2
                        )
                    )
                )
                + 1
            )
        else:
            half_length = particle_size // 2
        length = 2 * half_length
        assert len(positions) == len(orientations)
        if num_particles <= 0:
            num_particles = len(positions)
        else:
            num_particles = min(num_particles, len(positions))
        print(
            "Averaging %d %s particles with box size %d" % (num_particles, name, length)
        )

        # Create the average array
        half = np.zeros(shape=(2, length, length, length), dtype="float32")
        num = np.zeros(shape=(2,), dtype="float32")

        # Sort the positions and orientations by y
        positions, orientations = zip(
            *sorted(zip(positions, orientations), key=lambda x: x[0][1])
        )

        # Get the random indices
        indices = list(
            np.random.choice(range(len(positions)), size=num_particles, replace=False)
        )
        indices = [indices[: num_particles // 2], indices[num_particles // 2 :]]

        # Loop through all the particles
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for half_index, data in lazy_map(
                executor,
                _process_sub_tomo,
                _iterate_particles(
                    indices,
                    positions,
                    orientations,
                    centre,
                    size,
                    half_length,
                    half.shape,
                    voxel_size,
                    tomogram,
                ),
            ):
                # Add the contribution to the average
                half[half_index, :, :, :] += data
                num[half_index] += 1

        # Average the sub tomograms
        print("Averaging half 1 with %d particles" % num[0])
        print("Averaging half 2 with %d particles" % num[1])
        if num[0] > 0:
            half[0, :, :, :] = half[0, :, :, :] / num[0]
        if num[1] > 0:
            half[1, :, :, :] = half[1, :, :, :] / num[1]

        # Save the averaged data
        print("Saving half 1 to %s" % half_1_filename)
        handle = mrcfile.new(half_1_filename, overwrite=True)
        handle.set_data(half[0, :, :, :])
        handle.voxel_size = tomo_file.voxel_size
        print("Saving half 2 to %s" % half_2_filename)
        handle = mrcfile.new(half_2_filename, overwrite=True)
        handle.set_data(half[1, :, :, :])
        handle.voxel_size = tomo_file.voxel_size


@singledispatch
def average_all_particles(
    config_file,
    sample_file: str,
    rec_file: str,
    average_file: str,
    particle_size: int,
    num_particles: int,
):
    """
    Perform sub tomogram averaging

    Args:
        config_file: The input config filename
        sample_file: The sample filename
        rec_file: The reconstruction filename
        average_file: The particle average filename
        particle_size: The particle size (px)

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Load the sample
    sample = parakeet.sample.load(sample_file)

    # Do the sub tomogram averaging
    _average_all_particles_Config(
        config.scan, sample, rec_file, average_file, particle_size, num_particles
    )


@average_all_particles.register(parakeet.config.Scan)
def _average_all_particles_Config(
    config: parakeet.config.Scan,
    sample: parakeet.sample.Sample,
    rec_filename: str,
    average_filename: str,
    particle_size: int = 0,
    num_particles: int = 0,
):
    """
    Average particles to compute averaged reconstruction

    """

    # Get the scan config
    scan = config.model_dump()

    # Get the sample centre
    centre = np.array(sample.centre)

    # Read the reconstruction file
    tomo_file = mrcfile.mmap(rec_filename)
    tomogram = tomo_file.data

    # Get the size of the volume
    voxel_size = np.array(
        (
            tomo_file.voxel_size["x"],
            tomo_file.voxel_size["y"],
            tomo_file.voxel_size["z"],
        )
    )
    assert voxel_size[0] == voxel_size[1]
    assert voxel_size[0] == voxel_size[2]
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
                int(
                    ceil(
                        sqrt(
                            ((xmin - xc) / voxel_size[0]) ** 2
                            + ((ymin - yc) / voxel_size[1]) ** 2
                            + ((zmin - zc) / voxel_size[2]) ** 2
                        )
                    )
                )
                + 1
            )
        else:
            half_length = particle_size // 2
        length = 2 * half_length
        assert len(positions) == len(orientations)
        if num_particles <= 0:
            num_particles = len(positions)
        else:
            num_particles = min(num_particles, len(positions))
        print(
            "Averaging %d %s particles with box size %d" % (num_particles, name, length)
        )

        # Create the average array
        average = np.zeros(shape=(length, length, length), dtype="float32")
        num = 0.0

        # Sort the positions and orientations by y
        positions, orientations = zip(
            *sorted(zip(positions, orientations), key=lambda x: x[0][1])
        )

        # Get the random indices
        indices = [list(range(len(positions)))]

        # Loop through all the particles
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for half_index, data in lazy_map(
                executor,
                _process_sub_tomo,
                _iterate_particles(
                    indices,
                    positions,
                    orientations,
                    centre,
                    size,
                    half_length,
                    average.shape,
                    voxel_size,
                    tomogram,
                ),
            ):
                # Add the contribution to the average
                average += data
                num += 1
                print("Count: ", num)

        # Average the sub tomograms
        print("Averaging map with %d particles" % num)
        if num > 0:
            average = average / num

        # from matplotlib import pylab
        # pylab.imshow(average[half_length, :, :])
        # pylab.show()

        # Save the averaged data
        print("Saving map to %s" % average_filename)
        handle = mrcfile.new(average_filename, overwrite=True)
        handle.set_data(average)
        handle.voxel_size = tomo_file.voxel_size
