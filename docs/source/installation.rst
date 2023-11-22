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
privileges):


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

Environment variables
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
and you will receive CUDA errors when attempting to run the simulations on the
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

Parakeet can also be run via Docker
(https://www.docker.com/get-started). To pull the docker container with the latest version
parakeet, you can do the following:

.. code-block:: bash
  
  docker pull quay.io/rosalindfranklininstitute/parakeet:latest

You may also pull the docker container of a specific version of parakeet. To do this in the above
command replace the 'latest' with one of the tags that you can find here
( https://quay.io/repository/rosalindfranklininstitute/parakeet?tab=tags).

To use parakeet with docker with GPU support the host machine should have the
appropriate Nvidia drivers installed and docker needs to be installed with the
nvidia container toolkit
(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

To easily input and output data from the container the volume mechanism can be
used, where a workspace directory of the host machine is mounted to a directory
in the container (in the folder /mnt in the example below). For this reason it
is advised that all the relevant files (e.g. config.yaml, sample.h5, etc.)
should be present in the host workspace directory.

Below is an example on how to use parakeet after, its docker image has been pulled, to run parakeet commands:

.. code-block:: bash

  docker run --gpus all -v $(pwd):/mnt --workdir=/mnt parakeet:latest \
    parakeet.config.new

You may also pull and run parakeet with a single command.

.. code-block:: bash

  docker run --gpus all -v $(pwd):/mnt --workdir=/mnt \
    quay.io/rosalindfranklininstitute/parakeet:latest parakeet.config.new


Install as a Singularity image
------------------------------

Parakeet can also be run via Singularity
(https://sylabs.io/guides/2.6/user-guide/installation.html). To build
parakeet's singularity container you can do the following:

.. code-block:: bash

  singularity build parakeet.sif docker://quay.io/rosalindfranklininstitute/parakeet:latest

Again similar to docker, you may build the singularity container of a specific version of parakeet
by replacing the 'latest' in the command above with one of the tags that you can find here
( https://quay.io/repository/rosalindfranklininstitute/parakeet?tab=tags).

Also like with docker, to use parakeet with singularity and GPU support, the
host machine should have the appropriate Nvidia drivers installed.

Below is an example on how to use parakeet, after its singularity image has been build, to run parakeet commands:

.. code-block:: bash

  # You can open an interactive shell in the singularity container like this:
  singularity shell --nv --bind=/data/directory:/mnt --pwd=/mnt parakeet.sif

  # Or you can execute multiple commands
  singularity exec --nv --bind=/data/directory:/mnt --pwd=/mnt parakeet.sif \
    bash -c "parakeet.config.new && parakeet.sample.new"

You may also build and run parakeet with a single command.

.. code-block:: bash

  # You can open an interactive shell in the singularity container like this:
  singularity shell --nv --bind=/data/directory:/mnt --pwd=/mnt \
    docker://quay.io/rosalindfranklininstitute/parakeet:latest

  # Or you can execute multiple commands
  singularity exec --nv --bind=/data/directory:/mnt --pwd=/mnt \
    docker://quay.io/rosalindfranklininstitute/parakeet:latest \
    bash -c "parakeet.config.new && parakeet.sample.new"


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

  FROM quay.io/rosalindfranklininstitute/parakeet:latest

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

  singularity build myapp.sif docker-daemon://me/myapp:latest


Install as a Apptainer image
------------------------------

Parakeet can also be run via Apptainer. Apptainer is the new name for the Singularity
project after is official move to the Linux Foundation
(https://apptainer.org/news/community-announcement-20211130/).

You may build and run parakeet via an Apptainer container the same way you would
build and run parakeet via an Singularity container, but instead of the command
'singularity' the command 'apptainer' should be used. The arguments of the command
'apptainer' is the exact same with the command 'singularity'.


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


Install on Baskerville HPC (native)
-----------------------------------

In order to install parakeet on the Baskerville tier 2 supercomputer (https://www.baskerville.ac.uk/)
within a virtual python environment (https://docs.baskerville.ac.uk/self-install/),
start an interactive job as follows (you will need to know your
account-project code and qos to do this):

.. code-block:: bash
  
  srun --account=${ACCOUNT} --qos=${QOS} --gpus=1 --time=0:20:0 \
    --export=USER,HOME,PATH,TERM --pty /bin/bash

When the interactive job starts (you may need to wait until the requested computational resources are
available), execute the following commands to install parakeet:

.. code-block:: bash
   
  # Load required modules
  module purge
  module load baskerville
  module load CUDA/11.4
  module load FFTW
  module load Python/3
   
  # Create a virtual environment
  python -m venv --system-site-packages env
  source env/bin/activate
  python -m pip install pip --upgrade

  # Install parakeet
  python -m pip install python-parakeet

You can run parakeet on Baskerville non-interactively and interactively.

To run parakeet on Baskerville non-interactively (https://docs.baskerville.ac.uk/jobs/)
you need write a script called run.sh with the following contents (you will need to know
your account-project code, qos and how much time do you expect to take for your script to finish):

.. code-block:: bash

  #!/bin/bash
  #SBATCH --account=$ACCOUNT
  #SBATCH --qos=$QOS
  #SBATCH --gpus=1
  #SBATCH --time=$TIME

  # Load required modules
  module purge
  module load baskerville
  module load CUDA/11.4
  module load FFTW
  module load Python/3

  # Activate environment
  source env/bin/activate

  # Parakeet commands
  parakeet.run -c config.yaml

To submit the run.sh script and therefore run the parakeet simulations execute the following:

.. code-block:: bash

  sbatch run.sh

The benefit of non-interactive job submissions is that your job will be executed automatically,
when the requested computational resources are available and that you can submit and queue multiple
jobs.

To run parakeet interactively (https://docs.baskerville.ac.uk/interactive-jobs/), execute:

.. code-block:: bash

  srun --account=${ACCOUNT} --qos=${QOS} --gpus=1 --time=${TIME} \
    --export=USER,HOME,PATH,TERM --pty /bin/bash

When the interactive job starts (you may need to wait until the requested computational resources are
available), execute the following commands to install parakeet:

.. code-block:: bash

  # Load required modules
  module purge
  module load baskerville
  module load CUDA/11.4
  module load FFTW
  module load Python/3

  # Activate environment
  source env/bin/activate

  # Parakeet commands
  parakeet.run -c config.yaml

Install on Baskerville (apptainer)
------------------------------------

Alternatively, you may run parakeet on the baskerville tier 2 supercomputer with
apptainer (https://docs.baskerville.ac.uk/containerisation/).

To run parakeet on Baskerville non-interactively (https://docs.baskerville.ac.uk/jobs/)
you need write a script called run.sh with the following contents (you will need to know
your account-project code, qos and how much time do you expect to take for your script to finish):

.. code-block:: bash

  #!/bin/bash
  #SBATCH --account=$ACCOUNT
  #SBATCH --qos=$QOS
  #SBATCH --gpus=1
  #SBATCH --time=$TIME

  # Load required modules
  module purge
  module load baskerville
  module load CUDA/11.4
  module load FFTW
  module load Python/3

  export APPTAINER_CACHEDIR=/path/to/cache/directory/within/your/project/directory

  function container {
    apptainer exec --nv --bind=/path/to/data/directory/within/your/project/directory:/mnt --pwd=/mnt \
      docker://quay.io/rosalindfranklininstitute/parakeet:latest bash -c "$@"
  }

  # Parakeet commands
  container \
  "echo Below you can have multiple commands && \
  parakeet.run -c config.yaml"

To submit the run.sh script and therefore run the parakeet simulations execute the following:

.. code-block:: bash

  sbatch run.sh

The benefit of non-interactive job submissions is that your job will be executed automatically,
when the requested computational resources are available and that you can submit and queue multiple
jobs.

To run parakeet interactively (https://docs.baskerville.ac.uk/interactive-jobs/), execute:

.. code-block:: bash

  srun --account=${ACCOUNT} --qos=${QOS} --gpus=1 --time=${TIME} \
    --export=USER,HOME,PATH,TERM --pty /bin/bash

When the interactive job starts (you may need to wait until the requested computational resources are
available), execute the following commands to install parakeet:

.. code-block:: bash

  # Load required modules
  module purge
  module load baskerville
  module load CUDA/11.4
  module load FFTW
  module load Python/3

  export APPTAINER_CACHEDIR=/path/to/cache/directory/within/your/project/directory

  function container {
    apptainer exec --nv --bind=/path/to/data/directory/within/your/project/directory:/mnt --pwd=/mnt \
      docker://quay.io/rosalindfranklininstitute/parakeet:latest bash -c "$@"
  }

  # Parakeet commands
  container \
  "echo Below you can have multiple commands && \
  parakeet.run -c config.yaml"


Install on STFC Scarf
---------------------

In order to install parakeet on the scarf with singularity, start and
interactive job as follows:

.. code-block:: bash
  
  salloc --time=1:0:0

Now run the following commands:

.. code-block:: bash

  singularity build parakeet.sif docker://quay.io/rosalindfranklininstitute/parakeet:latest

Once you are happy, log out of the interactive node. To run parakeet on scarf
write a script called run.sh with the following contents:

.. code-block:: bash

  #!/bin/bash
  #SBATCH --gpus=1

  function container {
    singularity exec --nv --bind=/path/to/data/directory/within/your/project/directory:/mnt --pwd=/mnt \
      docker://quay.io/rosalindfranklininstitute/parakeet:latest bash -c "$@"
  }

  # Parakeet commands
  container \
  "echo Below you can have multiple commands && \
  parakeet.run -c config.yaml"

Then run the simulations as follows:

.. code-block:: bash

  sbatch run.sh


Testing
-------

To run the tests, follow the :ref:`Installation for developers` instructions
and then do the following command from within the source distribution:

.. code-block:: bash
  
  pytest

