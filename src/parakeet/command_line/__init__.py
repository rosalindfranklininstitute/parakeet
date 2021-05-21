#
# parakeet.command_line.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import argparse
import gemmi
import logging
import logging.config
import numpy
import random
import scipy.signal
import parakeet.io
import parakeet.config
import parakeet.sample

# Get the logger
logger = logging.getLogger(__name__)


def configure_logging():
    """
    Configure the logging

    """

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
            "handlers": {
                "stream": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "parakeet": {
                    "handlers": ["stream"],
                    "level": "DEBUG",
                    "propagate": True,
                }
            },
        }
    )


def rebin(data, shape):
    """
    Rebin a multidimensional array

    Args:
        data (array): The input array
        shape (tuple): The new shape

    """
    # Get pairs of (shape, bin factor) for each dimension
    factors = numpy.array([(d, c // d) for d, c in zip(shape, data.shape)])

    # Rebin the array
    for i in range(len(factors)):
        data = scipy.signal.decimate(data, factors[i][1], axis=i)
    # data = data.reshape(factors.flatten())
    # for i in range(len(shape)):
    #     data = data.sum(-1 * (i + 1))
    return data


def export(argv=None):
    """
    Convert the input file type to a different file type

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read a PDB file")

    # Add an argument for the filename
    parser.add_argument("filename", type=str, default=None, help="The input filename")

    # Add an argument for the filename
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        dest="output",
        help="The output filename",
    )
    parser.add_argument(
        "--rot90",
        type=bool,
        default=False,
        dest="rot90",
        help="Rotate the image 90deg counter clockwise",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--rotation_range",
        type=str,
        default=None,
        dest="rotation_range",
        help="Select a rotation range",
    )
    group.add_argument(
        "--select_images",
        type=str,
        default=None,
        dest="select_images",
        help="Select a range of images (start,stop,step)",
    )
    parser.add_argument(
        "--roi", type=str, default=None, dest="roi", help="Select a region of interest"
    )
    parser.add_argument(
        "--complex_mode",
        choices=[
            "complex",
            "real",
            "imaginary",
            "amplitude",
            "phase",
            "phase_unwrap",
            "square",
            "imaginary_square",
        ],
        default="complex",
        dest="complex_mode",
        help="How to treat complex numbers",
    )
    parser.add_argument(
        "--interlace",
        type=int,
        default=None,
        dest="interlace",
        help="Interlace the scan",
    )
    parser.add_argument(
        "--rebin", type=int, default=1, dest="rebin", help="The rebinned factor"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        dest="vmin",
        help="The minimum pixel value when exporting to an image",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        dest="vmax",
        help="The maximum pixel value when exporting to an image",
    )

    # Parse the arguments
    args = parser.parse_args(argv)

    # Configure some basic logging
    configure_logging()

    # Read the input
    logger.info(f"Reading data from {args.filename}")
    reader = parakeet.io.open(args.filename)

    # Get the shape and indices to read
    if args.select_images is not None:
        logger.info("Selecting image range %s" % args.select_images)
        item = tuple(map(int, args.select_images.split(",")))
        indices = list(range(*item))
    else:
        indices = list(range(reader.shape[0]))

    # Get the shape and indices to read
    if args.rotation_range is not None:
        args.rotation_range = tuple(map(int, args.rotation_range.split(",")))
        indices = []
        for i in range(reader.shape[0]):
            angle = reader.angle[i]
            if angle >= args.rotation_range[0] and angle < args.rotation_range[1]:
                indices.append(i)
            else:
                logger.info(f"    Skipping image {i} because angle is out of range")

    # Interlace the images
    if args.interlace is not None:
        if args.interlace >= 1:
            interlaced_indices = []
            for i in range(args.interlace):
                interlaced_indices.extend(
                    [indices[j] for j in range(i, len(indices), args.interlace)]
                )
            indices = interlaced_indices
        else:
            random.shuffle(indices)

    # Get the region of interest
    if args.roi is not None:
        x0, y0, x1, y1 = tuple(map(int, args.roi.split(",")))
        assert x1 > x0
        assert y1 > y0
    else:
        x0, y0, x1, y1 = 0, 0, reader.data.shape[2], reader.data.shape[1]

    # If squared and dtype is complex then change to float
    if args.complex_mode != "complex":
        dtype = "float64"
    else:
        dtype = reader.data.dtype.name

    # Set the dataset shape
    shape = (len(indices), y1 - y0, x1 - x0)

    # If rotating, then rotate shape
    if args.rot90:
        shape = (shape[0], shape[2], shape[1])

    # Rebin
    if args.rebin != 1:
        shape = (shape[0], shape[1] // args.rebin, shape[2] // args.rebin)
        pixel_size = reader.pixel_size * args.rebin
    else:
        pixel_size = reader.pixel_size

    # Create the write
    logger.info(f"Writing data to {args.output}")
    writer = parakeet.io.new(
        args.output, shape=shape, pixel_size=pixel_size, dtype=dtype
    )

    # If converting to images, determine min and max
    if writer.is_image_writer:
        if args.vmin is None or args.vmax is None:
            logger.info("Computing min and max of dataset:")
            min_image = []
            max_image = []
            for i in indices:

                # Transform if necessary
                image = {
                    "complex": lambda x: x,
                    "real": lambda x: numpy.real(x),
                    "imaginary": lambda x: numpy.imag(x),
                    "amplitude": lambda x: numpy.abs(x),
                    "phase": lambda x: numpy.real(numpy.angle(x)),
                    "phase_unwrap": lambda x: numpy.unwrap(numpy.real(numpy.angle(x))),
                    "square": lambda x: numpy.abs(x) ** 2,
                    "imaginary_square": lambda x: numpy.imag(x) ** 2 + 1,
                }[args.complex_mode](reader.data[i, y0:y1, x0:x1])

                min_image.append(numpy.min(image))
                max_image.append(numpy.max(image))
                logger.info(
                    "    Reading image %d: min/max: %.2f/%.2f"
                    % (i, min_image[-1], max_image[-1])
                )
            writer.vmin = min(min_image)
            writer.vmax = max(max_image)
            logger.info("Min: %f" % writer.vmin)
            logger.info("Max: %f" % writer.vmax)
        if args.vmin:
            writer.vmin = args.vmin
        if args.vmax:
            writer.vmax = args.vmax

    # Write the data
    for j, i in enumerate(indices):
        logger.info(f"    Copying image {i} -> image {j}")

        # Get the image info
        image = reader.data[i, y0:y1, x0:x1]
        angle = reader.angle[i]
        position = reader.position[i]

        # Rotate if necessary
        if args.rot90:
            image = numpy.rot90(image)
            position = (position[1], position[0], position[2])

        # Transform if necessary
        image = {
            "complex": lambda x: x,
            "real": lambda x: numpy.real(x),
            "imaginary": lambda x: numpy.imag(x),
            "amplitude": lambda x: numpy.abs(x),
            "phase": lambda x: numpy.real(numpy.angle(x)),
            "phase_unwrap": lambda x: numpy.unwrap(numpy.real(numpy.angle(x))),
            "square": lambda x: numpy.abs(x) ** 2,
            "imaginary_square": lambda x: numpy.imag(x) ** 2 + 1,
        }[args.complex_mode](image)

        # Rebin the array
        if args.rebin != 1:
            new_shape = numpy.array(image.shape) // args.rebin
            image = rebin(image, new_shape)

        # Write the image info
        writer.data[j, :, :] = image
        writer.angle[j] = angle
        writer.position[j] = position

    # Update the writer
    writer.update()


def read_pdb():
    """
    Read the given PDB file and show the atom positions

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read a PDB file")

    # Add an argument for the filename
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        dest="filename",
        help="The path to the PDB file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check a filename has been given
    if args.filename == None:
        parser.print_help()
        exit(0)

    # Configure some basic logging
    configure_logging()

    # Read the structure
    structure = gemmi.read_structure(args.filename)

    # Iterate through atoms
    prefix = " " * 4
    logger.info("Structure: %s" % structure.name)
    for model in structure:
        logger.info("%sModel: %s" % (prefix, model.name))
        for chain in model:
            logger.info("%sChain: %s" % (prefix * 2, chain.name))
            for residue in chain:
                logger.info("%sResidue: %s" % (prefix * 3, residue.name))
                for atom in residue:
                    logger.info(
                        "%sAtom: %s, %f, %f, %f, %f, %f, %f"
                        % (
                            prefix * 4,
                            atom.element.name,
                            atom.pos.x,
                            atom.pos.y,
                            atom.pos.z,
                            atom.occ,
                            atom.charge,
                            parakeet.sample.get_atom_sigma(atom),
                        )
                    )
