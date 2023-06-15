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


import logging
import logging.config
import numpy as np
import random
import parakeet.io
import parakeet.config
import parakeet.sample
from argparse import ArgumentParser
from typing import List


__all__ = ["export"]


# Get the logger
logger = logging.getLogger(__name__)


def rebin(data, shape, filter=True):
    """
    Rebin a multidimensional array

    Args:
        data (array): The input array
        shape (tuple): The new shape
        filter: Filter in Fourier space

    """
    if filter:
        f = np.fft.fft2(data)
        f = np.fft.fftshift(f)
        yc, xc = data.shape[0] // 2, data.shape[1] // 2
        yh = shape[0] // 2
        xh = shape[1] // 2
        x0 = xc - xh
        y0 = yc - yh
        x1 = x0 + shape[1]
        y1 = y0 + shape[0]
        y, x = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
        r = (y - yc) ** 2 / yh**2 + (x - xc) ** 2 / xh**2
        mask = r < 1.0
        f = f * mask
        f = f[y0:y1, x0:x1]
        f = np.fft.ifftshift(f)
        d = np.fft.ifft2(f)
        output = d.real
    else:
        shape = (
            shape[0],
            data.shape[0] // shape[0],
            shape[1],
            data.shape[1] // shape[1],
        )
        output = data.reshape(shape).sum(-1).sum(1)
    return output


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
    f = np.fft.fft2(data)
    f = np.fft.fftshift(f)
    yc, xc = data.shape[0] // 2, data.shape[1] // 2
    y, x = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    r = (
        np.sqrt((y - yc) ** 2 / data.shape[0] ** 2 + (x - xc) ** 2 / data.shape[1] ** 2)
        / pixel_size
    )
    if shape == "square":
        g = r < 1.0 / resolution
    elif shape == "guassian":
        g = np.exp(-0.5 * r**2 * resolution**2)
    f = f * g
    f = np.fft.ifftshift(f)
    d = np.fft.ifft2(f)
    return d.real


def get_description():
    """
    Get the program description

    """
    return "Export images to a different format"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for parakeet.export

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

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
        help=(
            "Select a rotation range (deg).\n"
            "\n"
            "Multiple rotation ranges can be specified as:\n"
            "--rotation_range=start1,stop1;start2,stop2"
        ),
    )
    group.add_argument(
        "--select_images",
        type=str,
        default=None,
        dest="select_images",
        help="Select a range of images (start,stop,step)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        dest="roi",
        help=("Select a region of interest (--roi=x0,y0,x1,y1)"),
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
        help=(
            "Interlace the scan. If the value <= 1 then the images are kept in "
            "the same order, otherwise, the images are reordered by skipping "
            "images. For example, if --interlace=2 is set then the images will "
            "be written out as [1, 3, 5, ... , 2, 4, 6, ...]"
        ),
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=1,
        dest="rebin",
        help="The rebinned factor. The shape of the output images will be original_shape / rebin",
    )
    parser.add_argument(
        "--filter_resolution",
        dest="filter_resolution",
        type=float,
        default=None,
        help="The resolution of the filter (A)",
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


def export_impl(args):
    """
    Convert the input file type to a different file type

    """
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
                    "real": lambda x: np.real(x),
                    "imaginary": lambda x: np.imag(x),
                    "amplitude": lambda x: np.abs(x),
                    "phase": lambda x: np.real(np.angle(x)),
                    "phase_unwrap": lambda x: np.unwrap(np.real(np.angle(x))),
                    "square": lambda x: np.abs(x) ** 2,
                    "imaginary_square": lambda x: np.imag(x) ** 2 + 1,
                }[args.complex_mode](reader.data[i, y0:y1, x0:x1])

                min_image.append(np.min(image))
                max_image.append(np.max(image))
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
        header = reader.header[i]

        # Rotate if necessary
        if args.rot90:
            image = np.rot90(image)
            header["shift_x"] = header["shift_y"]
            header["shift_y"] = header["shift_x"]
            header["stage_z"] = header["stage_z"]

        # Transform if necessary
        image = {
            "complex": lambda x: x,
            "real": lambda x: np.real(x),
            "imaginary": lambda x: np.imag(x),
            "amplitude": lambda x: np.abs(x),
            "phase": lambda x: np.real(np.angle(x)),
            "phase_unwrap": lambda x: np.unwrap(np.real(np.angle(x))),
            "square": lambda x: np.abs(x) ** 2,
            "imaginary_square": lambda x: np.imag(x) ** 2 + 1,
        }[args.complex_mode](image)

        # Filter the images
        if args.filter_shape is not None:
            image = filter_image(
                image, pixel_size, args.filter_resolution, args.filter_shape
            )

        # Rebin the array
        if args.rebin != 1:
            new_shape = np.array(image.shape) // args.rebin
            image = rebin(image, new_shape, filter=False)

        # Write the image info
        writer.data[j, :, :] = image
        writer.header[j] = header

    # Update the writer
    writer.update()


def export(args: List[str] = None):
    """
    Convert the input file type to a different file type

    """
    export_impl(get_parser().parse_args(args=args))
