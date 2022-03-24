#
# parakeet.analyse.refine.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import maptools
import random
import parakeet.sample


__all__ = ["refine"]


# Set the random seed
random.seed(0)


def refine(sample_filename: str, rec_filename: str):
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
