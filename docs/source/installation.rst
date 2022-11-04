Installation
============

Dependencies
------------

In order to build this package, the following dependencies are required:

- A C++ compiler (e.g. g++)
- FFTW
- Python development libraries
- The CUDA toolkit

On ubuntu 20.04, the dependencies can be install on a clean install as follows
(you may need to contact your system administrator if you do not have admin
priviliges):


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
  sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
  sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
  sudo apt-get update
  sudo apt-get -y install cuda

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

  # Clone the repository
  git clone https://github.com/rosalindfranklininstitute/parakeet.git

  # Enter the parakeet directory
  cd parakeet

  # Checkout the submodules
  git submodule update --init --recursive

  # Install the package locally
  pip install .


.. _Installation for developers:

Installation for developers
---------------------------

Run the following commands to install in development mode after setting the
environment variables described above:

.. code-block:: bash

  # Clone the repository
  git clone https://github.com/rosalindfranklininstitute/parakeet.git

  # Enter the parakeet directory
  cd parakeet

  # Checkout the submodules
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

  python -m pip install git+https://github.com/rosalindfranklininstitute/parakeet.git@master

Install using conda
-------------------

You can install parakeet using conda as follows:

.. code-block:: bash

  # Create a conda environment
  conda create -n parakeet python=3.9

  # Install parakeet
  conda install -c conda-forge -c james.parkhurst python-parakeet


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

Below is an example on how to use parakeet with docker to run parakeet commands:

.. code-block:: bash

  docker run --gpus all -v $(pwd):/mnt --workdir=/mnt parakeet:master \
    parakeet.config.new 


Install as a Singularity image
------------------------------

Parakeet can also be installed and used via Singularity
(https://sylabs.io/guides/2.6/user-guide/installation.html). To download
parakeet's singularity container you can do the following:

.. code-block:: bash

  singularity build parakeet.sif docker://ghcr.io/rosalindfranklininstitute/parakeet:master

Again similar to docker, to use parakeet with singularity and GPU support, the
host machine should have the approprate Nvidia drivers installed.

Below is an example on how to use parakeet with singularity to run parakeet commands:

.. code-block:: bash

  singularity run --nv parakeet.sif \
    parakeet.config.new


Install as Singularity sandbox
------------------------------

If you need to modify the singularity container for development purposes, it is
possible to build a parakeet sandbox as follows:

.. code-block:: bash

  singularity build --sandbox parakeet_sandbox/ parakeet.sif

The source code for parakeet resides in the parakeet_sandbox/apps/ directory.
You can then modify the python code in place and use `singularity shell` or
`singularity run` to install the changes as follows:

.. code-block:: bash

  singularity run --writable parakeet_sandbox/ pip install /app --prefix=/usr/local

Likewise, new software packages can be install into the container as follows:

.. code-block:: bash

  singularity run --writable parakeet_sandbox/ pip install ${PACKAGE} --prefix=/usr/local

To run parakeet from the sandbox, execute with the following command:

.. code-block:: bash

  singularity run --nv parakeet_sandbox/ parakeet.run -c config.yaml


If you want to rebuild the singularity image from the sandbox you can then do
the following:

.. code-block:: bash

  singularity build parakeet_image.sif parakeet_sandbox/


Build a derivative Singularity image
------------------------------------

You can build a new container depending on the parakeet docker container as
follows. In your python source code repository create a file called Dockerfile
with the following contents:

.. code-block:: bash

  FROM ghcr.io/rosalindfranklininstitute/parakeet:master

  WORKDIR /myapp
  COPY . .

  RUN apt update
  RUN git submodule update --init --recursive
  RUN pip install .

Now build locally with docker:

.. code-block:: bash

  sudo docker build . -t me/myapp

Now we can build the singularity image from the docker image

.. code-block:: bash

  singularity build myapp.sif docker-deamon://me/myapp:latest


Install as a snap
-----------------

You can install the parakeet snap from the snapcraft repository as follows:

.. code-block:: bash

  # Install the snap from the edge channel
  sudo snap install parakeet --classic --edge

You can also build the parakeet snap application from source as follows:

.. code-block:: bash

  # Run this command in the repository directory on a VM with 4GB memory
  SNAPCRAFT_BUILD_ENVIRONMENT_MEMORY=4G snapcraft

  # Install the locally built snap
  sudo snap install parakeet_${VERSION}.snap --classic --dangerous

Note that the snap installation only exposes the top level parakeet command:

.. code-block:: bash

  parakeet -h


Install on Baskerville (native)
-------------------------------

In order to install parakeet on the baskerville tier 2 supercomputer with
singularity, start an interactive job as follows (you will need to know your
account number and qos to do this):

.. code-block:: bash
  
  salloc --account=${ACCOUNT} --qos=${QOS} --gpus=1 --time=1:0:0
  srun --pty bash -i

Now execute the following commands to install parakeet:

.. code-block:: bash
   
  # Load required modules
  module purge
  module load baskerville
  module load bask-apps/test
  module load CUDA/11.4
  module load FFTW
  module load Python/3
   
  # Create a virtual environment
  python -m venv env
  source env/bin/activate
  python -m pip install pip --upgrade

  # Install parakeet
  python -m pip install python-parakeet

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
  module load bask-apps/test
  module load CUDA/11.4
  module load FFTW
  module load Python/3

  # Activate environment
  source env/bin/activate

  # Parakeet commands
  parakeet run -c config.yaml

Then run the simulations as follows:

.. code-block:: bash

  sbatch run.sh


Install on Baskerville (singularity)
------------------------------------

In order to install parakeet on the baskerville tier 2 supercomputer with
singularity, start an interactive job as follows (you will need to know your
account number and qos to do this):

.. code-block:: bash
  
  salloc --account=${ACCOUNT} --qos=${QOS} --gpus=1 --time=1:0:0
  srun --pty bash -i

Now run the following commands:

.. code-block:: bash

  # Load required modules
  module purge
  module load baskerville
  module load bask-singularity-conf/live

  # Install package
  singularity build parakeet.sif docker://ghcr.io/rosalindfranklininstitute/parakeet:master

Once you are happy, log out of the interactive node. To run parakeet on
baskerville write a script called run.sh with the following contents:

.. code-block:: bash

  #!/bin/bash
  #SBATCH --account=$ACCOUNT
  #SBATCH --qos=$QOS
  #SBATCH --gpus=1

  # Load required modules
  module purge
  module load baskerville
  module load bask-singularity-conf/live

  function parakeet {
    singularity run --nv parakeet.sif parakeet $@
  }

  # Parakeet commands
  parakeet run -c config.yaml

Then run the simulations as follows:

.. code-block:: bash

  sbatch run.sh

.. warning::
  
  On Baskerville, you may receive an error like this when using the parakeet sandbox:

.. code-block:: bash

  WARNING: By using --writable, Singularity can't create /bask destination automatically without overlay or underlay            
  FATAL:   container creation failed: mount /var/singularity/mnt/session/bask->/bask error: while mounting /var/singularity/mnt/
  session/bask: destination /bask doesn't exist in container

You can fix this by creating the following directory within the sandbox directory:

.. code-block:: bash

  mkdir -p parakeet_sandbox/bask


Install on STFC Scarf
---------------------

In order to install parakeet on the scarf with singularity, start and
interactive job as follows:

.. code-block:: bash
  
  salloc --time=1:0:0

Now run the following commands:

.. code-block:: bash

  singularity build parakeet.sif docker://ghcr.io/rosalindfranklininstitute/parakeet:master

Once you are happy, log out of the interactive node. To run parakeet on scarf
write a script called run.sh with the following contents:

.. code-block:: bash

  #!/bin/bash
  #SBATCH --gpus=1

  function parakeet {
    singularity run --nv parakeet.sif parakeet $@
  }

  # Parakeet commands
  parakeet run -c config.yaml

Then run the simulations as follows:

.. code-block:: bash

  sbatch run.sh


Testing
-------

To run the tests, follow the :ref:`Installation for developers` instructions
and then do the following command from within the source distribution:

.. code-block:: bash
  
  pytest

