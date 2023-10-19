#
# parakeet.futures.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import concurrent.futures
import multiprocessing
import parakeet.config


def as_completed(futures):
    return concurrent.futures.as_completed(futures)
    # import dask.distributed
    # return dask.distributed.as_completed(futures)


def factory(method="mp", max_workers=1):
    """
    Configure the future to use for parallel processing

    Args:
        method (str): The cluster method (sge)
        max_workers (int): The number of worker processes

    """

    if method == "sge":
        import dask_jobqueue
        import dask.distributed

        # Create the SGECluster object
        # For each worker:
        #   - request 1 gpu and 1 core
        #   - use 1 process per worker and 1 thread per process
        # This ensures that only 1 job will be run on each worker at any time
        # which is required to ensure that there is not any competition for
        # gpu resources (by default dask will try to run many jobs at the same
        # time which causes errors and instability).
        cluster = dask_jobqueue.SGECluster(
            cores=1,
            processes=1,
            nthreads=1,
            memory="64 GB",
            queue="all.q",
            project="tomography",
            resource_spec="gpu=1",
            job_extra=["-V"],
            name="parakeet",
            local_directory=parakeet.config.temp_directory(),
            log_directory=parakeet.config.temp_directory(),
        )

        # Set the number of worker nodes
        cluster.scale(max_workers)

        # Return the client
        executor = dask.distributed.Client(cluster)
    elif method == "mp":
        # We set a process index for each process so that we always uses the
        # same GPU with the same process
        def initializer(count):
            global process_index
            process_index = count.value
            count.value += 1

        # Create the process pool executor
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializer,
            initargs=(multiprocessing.Value("i", 0),),
        )
    else:
        raise RuntimeError(f"Unknown multiprocessing method: {method}")

    # Return the executor
    return executor
