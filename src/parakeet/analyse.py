#
# parakeet.analyse.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy as np
import maptools
import mrcfile
import os.path
import guanaco
import random
import h5py
import scipy.ndimage
import scipy.spatial.transform
import parakeet.sample
from math import sqrt, ceil

# Set the random seed
random.seed(0)


def reconstruct(image_filename, rec_filename, microscope, simulation, device="gpu"):
    """
    Reconstruct the volume and use 3D CTF correction beforehand if the input image is uncorrected

    """

    # Ensure mrc file
    assert os.path.splitext(image_filename)[1] == ".mrc"

    # Set the corrected filename
    corrected_filename = os.path.join(os.path.dirname(rec_filename), "CORRECTED.dat")

    # Get the parameters for the CTF correction
    nx = microscope.detector.nx
    pixel_size = microscope.detector.pixel_size
    energy = microscope.beam.energy
    defocus = -microscope.lens.c_10
    num_defocus = int((nx * pixel_size) / 100)

    # Set the spherical aberration
    if simulation["inelastic_model"] == "cc_corrected":
        print("Setting spherical aberration to zero")
        spherical_aberration = 0
    else:
        spherical_aberration = microscope.lens.c_30

    astigmatism = microscope.lens.c_12
    astigmatism_angle = microscope.lens.phi_12
    phase_shift = 0

    # Do the reconstruction
    guanaco.reconstruct_file(
        input_filename=image_filename,
        output_filename=rec_filename,
        corrected_filename=corrected_filename,
        centre=None,
        energy=energy,
        defocus=defocus,
        num_defocus=num_defocus,
        spherical_aberration=spherical_aberration,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        phase_shift=phase_shift,
        angular_weights=True,
        device=device,
    )


def correct(
    image_filename,
    corrected_filename,
    microscope,
    simulation,
    num_defocus=None,
    device="gpu",
):
    """
    Correct the images using 3D CTF correction

    """

    # Ensure mrc file
    assert os.path.splitext(image_filename)[1] == ".mrc"

    # Get the parameters for the CTF correction
    nx = microscope.detector.nx
    pixel_size = microscope.detector.pixel_size
    energy = microscope.beam.energy
    defocus = -microscope.lens.c_10
    if num_defocus is None:
        num_defocus = int((nx * pixel_size) / 100)

    # Set the spherical aberration
    if simulation["inelastic_model"] == "cc_corrected":
        print("Setting spherical aberration to zero")
        spherical_aberration = 0
    else:
        spherical_aberration = microscope.lens.c_30

    astigmatism = microscope.lens.c_12
    astigmatism_angle = microscope.lens.phi_12
    phase_shift = 0

    # Do the reconstruction
    guanaco.correct_file(
        input_filename=image_filename,
        output_filename=corrected_filename,
        centre=None,
        energy=energy,
        defocus=defocus,
        num_defocus=num_defocus,
        spherical_aberration=spherical_aberration,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        phase_shift=phase_shift,
        device=device,
    )


def average_particles(
    scan,
    sample_filename,
    rec_filename,
    half_1_filename,
    half_2_filename,
    particle_size=0,
    num_particles=0,
):
    """
    Average particles to compute averaged reconstruction

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

    # Load the sample
    sample = parakeet.sample.load(sample_filename)

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
        if num_particles <= 0:
            num_particles = len(positions)
        print(
            "Averaging %d %s particles with box size %d" % (num_particles, name, length)
        )

        # Create the average array
        half_1 = np.zeros(shape=(length, length, length), dtype="float32")
        half_2 = np.zeros(shape=(length, length, length), dtype="float32")
        num_1 = 0
        num_2 = 0

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
            if sub_tomo.shape == half_1.shape:

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
                if bool(random.getrandbits(1)):
                    half_1 += data
                    num_1 += 1
                else:
                    half_2 += data
                    num_2 += 1

            # Break if we have enough particles
            if num_1 + num_2 >= num_particles:
                break

        # Average the sub tomograms
        print("Averaging half 1 with %d particles" % num_1)
        print("Averaging half 2 with %d particles" % num_2)
        if num_1 > 0:
            half_1 = half_1 / num_1
        if num_2 > 0:
            half_2 = half_2 / num_2

        # from matplotlib import pylab
        # pylab.imshow(average[half_length, :, :])
        # pylab.show()

        # Save the averaged data
        print("Saving half 1 to %s" % half_1_filename)
        handle = mrcfile.new(half_1_filename, overwrite=True)
        handle.set_data(half_1)
        handle.voxel_size = tomo_file.voxel_size
        print("Saving half 2 to %s" % half_2_filename)
        handle = mrcfile.new(half_2_filename, overwrite=True)
        handle.set_data(half_2)
        handle.voxel_size = tomo_file.voxel_size


def average_all_particles(
    scan, sample_filename, rec_filename, average_filename, particle_size=0
):
    """
    Average particles to compute averaged reconstruction

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

    # Load the sample
    sample = parakeet.sample.load(sample_filename)

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
        average = np.zeros(shape=(length, length, length), dtype="float32")
        num = 0

        # Sort the positions and orientations by y
        positions, orientations = zip(
            *sorted(zip(positions, orientations), key=lambda x: x[0][1])
        )

        # Loop through all the particles
        for i, (position, orientation) in enumerate(zip(positions, orientations)):

            # Compute p within the volume
            start_position = np.array([0, scan["start_pos"], 0])
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
            if sub_tomo.shape == average.shape:

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

                average += data
                num += 1

        # Average the sub tomograms
        print("Averaging map with %d particles" % num)
        average = average / num

        # from matplotlib import pylab
        # pylab.imshow(average[half_length, :, :])
        # pylab.show()

        # Save the averaged data
        print("Saving map to %s" % average_filename)
        handle = mrcfile.new(average_filename, overwrite=True)
        handle.set_data(average)
        handle.voxel_size = tomo_file.voxel_size


def extract_particles(
    scan, sample_filename, rec_filename, extract_filename, particle_size=0
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

    # Load the sample
    sample = parakeet.sample.load(sample_filename)

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
        extract_map = []
        particle_instance = np.zeros(shape=(length, length, length), dtype="float32")
        num = 0

        # Sort the positions and orientations by y
        positions, orientations = zip(
            *sorted(zip(positions, orientations), key=lambda x: x[0][1])
        )

        # Loop through all the particles
        for i, (position, orientation) in enumerate(zip(positions, orientations)):

            # Compute p within the volume
            start_position = np.array([0, scan["start_pos"], 0])
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
        print("Saving extracted particles to %s" % extract_filename)
        handle = h5py.File(extract_filename, "w")
        data_handle = handle.create_dataset("data", extract_map.shape, chunks=True)
        data_handle[:] = extract_map[:]
        handle.close()


def refine(sample_filename, rec_filename):
    """
    Refine the molecule against the map

    """

    # Load the sample
    sample = parakeet.sample.load(sample_filename)

    # Get the molecule name
    assert sample.number_of_molecules == 1
    name, _ = list(sample.iter_molecules())[0]

    # Get the PDB filename
    pdb_filename = parakeet.data.get_pdb(name)

    # Fit the molecule to the map
    maptools.fit(
        rec_filename,
        pdb_filename,
        output_pdb_filename="refined.pdb",
        resolution=3,
        ncycle=10,
        mode="rigid_body",
        log_filename="fit.log",
    )
