#
# elfantasma.futures.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import dask.distributed
import dask_jobqueue
import elfantasma.config


def factory(method="sge", max_workers=1):
    """
    Configure the future to use for parallel processing

    Args:
        method (str): The cluster method (sge)
        max_workers (int): The number of worker processes

    """
    if method == "sge":

        # Create the SGECluster object
        cluster = dask_jobqueue.SGECluster(
            cores=8,
            memory="64 GB",
            queue="all.q",
            project="tomography",
            resource_spec="gpu=1",
            job_extra=["-V"],
            name="elfantasma",
            local_directory=elfantasma.config.temp_directory(),
            log_directory=elfantasma.config.temp_directory(),
        )

        # Set the number of worker nodes
        cluster.scale(max_workers)

        # Return the client
        executor = dask.distributed.Client(cluster)
    else:
        raise RuntimeError(f"Unknown multiprocessing method: {method}")

    # Return the executor
    return executor
