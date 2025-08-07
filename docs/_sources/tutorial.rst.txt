Tutorial
========

Simulation of datasets is split into a number of different commands. Each
command takes a set of command line options or a configuration file in YAML
format.

Generate a new configuration file
---------------------------------

.. code-block:: bash

  parakeet.config.new

This will generate a new file called config.yaml which will contain the
following configuration.

.. code-block:: yaml

  microscope:
    beam:
      electrons_per_angstrom: 30
      energy: 300
      source_spread: 0.1
    detector:
      nx: 1000
      ny: 1000
      pixel_size: 1
    lens:
      c_10: -20000
      c_30: 2.7
      c_c: 2.7
  sample:
    box:
    - 1000
    - 1000
    - 1000
    centre:
    - 500
    - 500
    - 500
    molecules: null
    shape:
      cube:
        length: 1000.0
      cuboid:
        length_x: 1000.0
        length_y: 1000.0
        length_z: 1000.0
      cylinder:
        length: 1000.0
        radius: 500.0
      margin:
      - 0
      - 0
      - 0
      type: cube
  scan:
    mode: still
    num_images: 1
    start_angle: 0
    step_angle: 0
  simulation:
    ice: false

Edit the configuration file
---------------------------

The configuration file now needs to be edited to perform the desired
simulation. For example, to simulate with a single molecule we modify the
sample.molecules field as follows.

.. code-block:: yaml

  microscope:
    beam:
      electrons_per_angstrom: 30
      energy: 300
      source_spread: 0.1
    detector:
      nx: 1000
      ny: 1000
      pixel_size: 1
    lens:
      c_10: -20000
      c_30: 2.7
      c_c: 2.7
  sample:
    box:
    - 1000
    - 1000
    - 1000
    centre:
    - 500
    - 500
    - 500
    molecules:
      pdb:
        - id: 4v1w
          instances: 1
    shape:
      cube:
        length: 1000.0
      cuboid:
        length_x: 1000.0
        length_y: 1000.0
        length_z: 1000.0
      cylinder:
        length: 1000.0
        radius: 500.0
      margin:
      - 0
      - 0
      - 0
      type: cube
  scan:
    mode: still
    num_images: 1
    start_angle: 0
    step_angle: 0
  simulation:
    ice: false

Generate sample model
---------------------

Once the configuration file has been generated a new sample file can be created
with the following command:

.. code-block:: bash

  parakeet.sample.new -c config.yaml


This will result in a file "sample.h5" being generated. This file contains
information about the size and shape of the sample but as yet doesn't contain
any atomic coordinates. The atomic model is added by running the following
command which adds molecules to the sample file. If a single molcule is
specified then it will be placed in the centre of the sample volume. If
multiple molecules are specified then the molecules will be positioned at
random locations in the sample volume. This command will update the "sample.h5"
file with the atomic coordinates but will not generated any new files.

.. code-block:: bash

  parakeet.sample.add_molecules -c config.yaml


Simulate EM images
------------------

Once the atomic model is ready, the EM images can be simulated with the
following commands. Each stage of the simulation is separated because it may be
desirable to simulate many different defocused images from the sample exit wave
for example or many different doses for the sample defocusses image. Being
separate, the output of one stage can be reused for multiple runs of the next
stage. The first stage is to simulate the exit wave. This is the propagation of
the electron wave through the sample. It is therefore the most computationally
intensive part of the processes since the contribution of all atoms within the
sample needs to be calculated.


.. code-block:: bash

  parakeet.simulate.exit_wave -c config.yaml


This command will generate a file "exit_wave.h5" which will contain the exit
wave of all tilt angles. The next step is to simulate the micropscope optics
which is done with the following command:

.. code-block:: bash

  parakeet.simulate.optics -c config.yaml


This step is much quicker as it only scales with the size of the detector image
and doesn't require the atomic coordinates again. The command will output a
file "optics.h5". Finally, the response of the detector can be simulated with
the following command:

.. code-block:: bash

  parakeet.simulate.image -c config.yaml


This command will add the detector DQE and the Poisson noise for a given dose
and will output a file "image.h5".

Export functions
----------------

Typically we cant to output an MRC file for further processing. The hdf5 files
can easily be exported to MRC by the following command:

.. code-block:: bash

  parakeet.export file.h5 -o file.mrc
  
The export command can also be used to rebin the image or select a region of interest. 

Analysis functions
------------------

The reconstruction function can be called in Parakeet in the following way:

.. code-block:: bash

   parakeet.export image.h5 -o image.mrc
   parakeet.analyse.reconstruct -c config.yaml -i image.mrc

Averaging of the particles can then be done with:

.. code-block:: bash

   parakeet.analyse.average_particles -c config.yaml
