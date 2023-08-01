#
# parakeet.data.__init__.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import os.path
import profet
import warnings


def get_remote_pdb(pdb_id: str) -> tuple:
    """
    Get the pdb from a remote source using profet

    """

    # Get the fetcher
    fetcher = profet.Fetcher()

    # Download the data
    filedata = None
    filename = None
    for filetype in ["cif", "pdb"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filename, filedata = fetcher.get_file(pdb_id, filetype=filetype)
            if filedata is not None:
                filename = "%s.%s" % (filename, filetype)
                break
    else:
        raise RuntimeError("No PDB or CIF file found")

    if not isinstance(filedata, str):
        filedata = filedata.decode("ascii")

    # Return filedata
    return filename, filedata


def get_and_save_remote_pdb(pdb_id: str, directory: str) -> str:
    """
    Get the pdb and save it to file

    """
    # Get the filename and filedata
    filename, filedata = get_remote_pdb(pdb_id)

    # Construct the path
    filepath = os.path.join(directory, filename)

    # Save the file
    if filepath is not None and filedata is not None:
        with open(filepath, "w") as outfile:
            outfile.write(filedata)

    # Return the file path
    return filepath


def get_local_path() -> str:
    """
    Get the local pdb file directory

    """
    return os.path.join(os.path.dirname(__file__), "files")


def get_cache_path() -> str:
    """
    Get the cache pdb file directory

    """
    return os.path.expanduser(
        os.getenv("PARAKEET_CACHE", os.path.join("~", ".cache", "parakeet"))
    )


def get_pdb_cache() -> dict:
    """
    Get the PDB cache dictionary

    """

    cache = {}

    # Check local and cache directories
    for directory in [get_local_path(), get_cache_path()]:
        # Create the directory if not exists
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Check all the files in the directory for CIF and PDB files
        for filename in os.listdir(directory):
            if filename.endswith(".cif") or filename.endswith(".pdb"):
                pdb_id = os.path.splitext(filename)[0]
                cache[pdb_id] = os.path.join(directory, filename)

    return cache


def get_pdb(name):
    """
    Return the path to the data file

    Args:
        name (str): The pdb id

    Returns:
        str: The absolute filename

    """
    cache = get_pdb_cache()
    if name not in cache:
        filename = get_and_save_remote_pdb(name, get_cache_path())
        cache[name] = filename
    return cache[name]


def get_path(name):
    """
    Return the path to the data file

    Args:
        name (str): The relative filename

    Returns:
        str: The absolute filename

    """
    return os.path.join(get_local_path(), name)


def get_4v1w():
    """
    Get the path to the file containing the structure of Apoferratin (4v1w)

    Returns:
        str: The abolsute path to 4v1w

    """
    return get_path("4v1w.cif")


def get_4v5d():
    """
    Get the path to the file containing the structure of ribosome (4v5d)

    Returns:
        str: The abolsute path to 4v5d

    """
    return get_path("4v5d.cif")


def get_6qt9():
    """
    Get the path to the file containing the structure of the icosohedral virus
    SH1 (6qt9)

    Returns:
        str: The abolsute path to 6qt9

    """
    return get_path("6qt9.cif")
