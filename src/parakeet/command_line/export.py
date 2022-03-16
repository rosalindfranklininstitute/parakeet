#
# parakeet.command_line.export.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import argparse
import logging
import logging.config
import numpy
import random
import parakeet.io
import parakeet.config
import parakeet.sample

# Get the logger
logger = logging.getLogger(__name__)


def rebin(data, shape):
    """
    Rebin a multidimensional array

    Args:
        data (array): The input array
        shape (tuple): The new shape

    """
    f = numpy.fft.fft2(data)
    f = numpy.fft.fftshift(f)
    yc, xc = data.shape[0] // 2, data.shape[1] // 2
    yh = shape[0] // 2
    xh = shape[1] // 2
    x0 = xc - xh
    y0 = yc - yh
    x1 = x0 + shape[1]
    y1 = y0 + shape[0]
    y, x = numpy.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    r = (y - yc) ** 2 / yh**2 + (x - xc) ** 2 / xh**2
    mask = r < 1.0
    f = f * mask
    f = f[y0:y1, x0:x1]
    f = numpy.fft.ifftshift(f)
    d = numpy.fft.ifft2(f)
    return d.real


def filter_image(data, pixel_size, resolution, shape):
    """
    Filter the image

    Args:
        data (array): The input array
        pixel_size (float): The pixel size (A)
        resolution (float): The filter resolution
        shape (str): The filter shape

    """
    if pixel_size == 0:
        pixel_size = 1
    f = numpy.fft.fft2(data)
    f = numpy.fft.fftshift(f)
    yc, xc = data.shape[0] // 2, data.shape[1] // 2
    y, x = numpy.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    r = (
        numpy.sqrt(
            (y - yc) ** 2 / data.shape[0] ** 2 + (x - xc) ** 2 / data.shape[1] ** 2
        )
        / pixel_size
    )
    if shape == "square":
        g = r < 1.0 / resolution
    elif shape == "guassian":
        g = numpy.exp(-0.5 * r**2 * resolution**2)
    f = f * g
    f = numpy.fft.ifftshift(f)
    d = numpy.fft.ifft2(f)
    return d.real


def get_parser():
    """
    Get the parser for parakeet.export

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
        "--filter_resolution",
        dest="filter_resolution",
        type=float,
        default=None,
        help="The resolution",
    )
    parser.add_argument(
        "--filter_shape",
        dest="filter_shape",
        type=str,
        choices=["square", "gaussian"],
        default=None,
        help="The shape of the filter",
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
    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        dest="sort",
        choices=["angle"],
        help="Sort the images",
    )

    return parser


def export(argv=None):
    """
    Convert the input file type to a different file type

    """
    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(argv)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

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
        args.rotation_range = args.rotation_range.split(";")
        indices = []
        for p in range(0, len(args.rotation_range)):
            current_rotation_range = tuple(map(int, args.rotation_range[p].split(",")))
            for i in range(reader.shape[0]):
                angle = reader.angle[i]
                if (
                    angle >= current_rotation_range[0]
                    and angle < current_rotation_range[1]
                ):
                    indices.append(i)
                    logger.info(f"    Image {i} added as within the rotation range(s)")
        indices = list(set(indices))
        indices.sort()

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

    # Sort the images
    if args.sort is not None:
        if args.sort == "angle":
            indices = sorted(indices, key=lambda i: reader.angle[indices[i]])

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
        pixel_size = reader.pixel_size
        drift = reader.drift[i]
        defocus = reader.defocus[i]

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

        # Filter the images
        if args.filter_shape is not None:
            image = filter_image(
                image, pixel_size, args.filter_resolution, args.filter_shape
            )

        # Rebin the array
        if args.rebin != 1:
            new_shape = numpy.array(image.shape) // args.rebin
            image = rebin(image, new_shape)

        # Write the image info
        writer.data[j, :, :] = image
        writer.angle[j] = angle
        writer.position[j] = position
        if drift is not None:
            writer.drift[j] = drift
        if defocus is not None:
            writer.defocus[j] = defocus

    # Update the writer
    writer.update()
