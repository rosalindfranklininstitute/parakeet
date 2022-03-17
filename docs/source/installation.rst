Installation
============

Dependencies
------------

In order to build this package, the following dependencies are required:

- A C++ compiler (e.g. g++)
- FFTW
- Python development libraries
- The CUDA toolkit

On ubuntu 20.04, the dependencies can be install on a clean install as follows:


.. code-block:: bash
  
  # Install GCC
  sudo apt-get install build-essential
  
  # Install FFTW
  sudo apt-get install libfftw3-dev

  # Install Python headers
  sudo apt-get install python3.8-dev
  sudo apt-get install python3.8-venv 
  
  # Install CUDA toolkit
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
  sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
  sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
  sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
  sudo apt-get updatesudo apt-get -y install cuda

Enviornment variables
---------------------

If you have multiple compiler versions or the compilers you want to use are not
automatically picked up by cmake, you can explicitly state the compiler
versions you would like to use as follows, where in this case we are using gcc
as the C++ compiler:

.. code-block:: bash

  export CXX=/usr/bin/g++
  export CUDACXX=/usr/local/cuda-11/bin/nvcc
  export FFTWDIR=/usr/local/

Depending on your GPU and the version of the CUDA toolkit you are using, it may
also be necessary to set the CMAKE_CUDA_ARCHITECTURES variable. This variable
is by default set to "OFF" in the CMakeLists.txt file which has the effect of
compiling CUDA kernels on the fly. If you have an old GPU, this may not work
and you will receive CUDA errors when attemping to run the simulations on the
GPU. In this case simply set the variable to the architecture supported by your
GPU as follows (the example below is for the compute_37 architecture):

.. code-block:: bash
  
  export CMAKE_CUDA_ARCHITECTURES=37


Python virtual environments
---------------------------

Before installing parakeet, you may want to setup a python virtual environment
in which to install it rather than installing into your system library. Use the
following commands before following the instructions below.

.. code-block:: bash

  python3 -m venv env
  source env/bin/activate


Install from source
-------------------

To install from source you can run the following commands after setting the
environment variables described above.

.. code-block:: bash

  # Close the repository and submodules
  git clone https://github.com/rosalindfranklininstitute/amplus-digital-twin.git
  git submodule update --init --recursive

  # Install the package locally
  pip install .


Installation for developers
---------------------------

Run the following commands to install in development mode after setting the
environment variables described above:

.. code-block:: bash

  # Close the repository and submodules
  git clone https://github.com/rosalindfranklininstitute/amplus-digital-twin.git
  git submodule update --init --recursive

  # Install the package locally
  pip install . -e


Install using PIP
-----------------

You can install parakeet from the python package archive using pip by running
the following command. This is a source package which needs to be built on your
local machine so the environment variables described above for CUDA, FFTW and
CXX may need to be set.

.. code-block:: bash

  pip install python-parakeet

It is also possible to install the version of parakeet on the master branch (or
any other branch) directly using pip by using the following command:

.. code-block:: bash

  python -m pip install git+https://github.com/rosalindfranklininstitute/amplus-digital-twin.git@master

Install using conda
-------------------

You can install parakeet using conda as follows:

.. code-block:: bash

  # Create a conda environment
  conda create -n parakeet python=3.9

  # Install parakeet
  conda install -c james.parkhurst python-parakeet


Install as a Docker container
-----------------------------

Parakeet can also be installed and used via Docker
(https://www.docker.com/get-started). To download parakeet's docker container
you can do the following:

.. code-block:: bash
  
  docker pull ghcr.io/rosalindfranklininstitute/parakeet:master

To use parakeet with docker with GPU support the host machine should have the
approprate Nvidia drivers installed and docker needs to be installed with the
nvidia container toolkit
(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

To easily input and output data from the container the volume mechanism can be
used, where a workspace directory of the host machine is mounted to a directory
in the container (in the folder /mnt in the example below). For this reason it
is advised that all the relevent files (e.g. config.yaml, sample.h5, etc.)
should be present in the host workspace directory.

Below is an example on how to use parakeet with docker to simulate the exit
wave:

.. code-block:: bash

  docker run --gpus all -v $(pwd):/mnt --workdir=/mnt parakeet:master \
    parakeet.simulate.exit_wave \
      -c config.yaml \
      -d gpu \
      -s sample.h5 \
      -e exit_wave.h5


Install as a Singularity image
------------------------------

Parakeet can also be installed and used via Singularity
(https://sylabs.io/guides/2.6/user-guide/installation.html). To download
parakeet's singularity container you can do the following:

.. code-block:: bash

  singularity build parakeet.sif docker://gchr.io/rosalindfranklininstitute/parakeet:master

Again similar to docker, to use parakeet with singularity and GPU support, the
host machine should have the approprate Nvidia drivers installed.

Below is an example on how to use parakeet with singularity to simulate the
exit wave:

.. code-block:: bash

  singularity run --nv parakeet.sif \
    parakeet.simulate.exit_wave \
      -c config_new.yaml \
      -d gpu \
      -s sample.h5 \
      -e exit_wave.h5


Install as a snap package
-------------------------

Parakeet is also packaged as a snap package which can be downlowded from from
the github page. The package can then be installed by running the following
command. 

.. code-block:: bash

  sudo snap install parakeet_0.2.7.snap --dangerous --classic


Install on Baskerville
----------------------

In order to install parakeet on the baskerville tier 2 supercomputer, write a
script called "install.sh" with the following contents.

.. code-block:: bash

  #!/bin/bash
  #SBATCH --account=$ACCOUNT
  #SBATCH --qos=$QOS
  #SBATCH --gpus=1

  # Load required modules
  module purge
  module load baskerville
  module load CUDA
  module load HDF5
  module load FFTW

  # Create environment
  #python3 -m venv env

  # Activate environment
  source env/bin/activate
  python -m pip install --upgrade pip

  # Install package
  python -m pip install git+https://github.com/rosalindfranklininstitute/amplus-digital-twin.git@master

You will need an account number and qos to do this. Then run the script using
the following command:

.. code-block:: bash

  sbatch install.sh

To run parakeet on a baskerville then write a script called run.sh with the
following contents:

.. code-block:: bash

  #!/bin/bash
  #SBATCH --account=$ACCOUNT
  #SBATCH --qos=$QOS
  #SBATCH --gpus=1

  # Load required modules
  module purge
  module load baskerville
  module load CUDA
  module load HDF5
  module load FFTW

  # Activate environment
  source env/bin/activate

  # Parakeet commands
  ...

Then run the simulations as follows:

.. code-block:: bash

  sbatch run.sh


Testing
-------

To run the tests, follow the installation instructions for developers and then
do the following command from within the source distribution:

.. code-block:: bash
  
  pytest

