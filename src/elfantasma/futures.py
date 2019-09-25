#
# elfantasma.futures.py
#
# Copyright (C) 2019 Diamond Light Source
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import dask.distributed
import dask_jobqueue
import concurrent.futures


def factory(mp_method="multiprocessing", max_workers=1):
    """
    Configure the future to use for parallel processing

    Args:
        method (str): The multiprocessing method (multiprocessing, sge)
        max_workers (int): The number of worker processes

    """
    if mp_method == "multiprocessing":
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    elif mp_method == "sge":

        # Create the SGECluster object
        cluster = dask_jobqueue.SGECluster(
            cores=1,
            memory="16 GB",
            queue="all.q",
            project="tomography",
            resource_spec="gpu=1",
            job_extra=["-V"],
            name="elfantasma",
            local_directory="_cluster",
            log_directory="_cluster",
        )

        # Set the number of worker nodes
        cluster.scale(max_workers)

        # Return the client
        executor = dask.distributed.Client(cluster)
    else:
        raise RuntimeError(f"Unknown multiprocessing method: {mp_method}")

    # Return the executor
    return executor
