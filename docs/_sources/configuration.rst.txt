Configuration
=============

Definitions
-----------

Parakeet is configured via a YAML configuration file. The parameters of the
configuration file are defined below and additional example configuation files
can be seen at the bottom of the page.

.. pydantic:: parakeet.config.Config

The default configuration parameters can be seen by typing the following
command:

Examples
--------

Basic configuration
^^^^^^^^^^^^^^^^^^^

This is the default configuration file as output by parakeet.config.new. This configuration file only shows the most useful parameters which you should set.

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

Full configuration
^^^^^^^^^^^^^^^^^^

The full configuration is somewhat longer and contains parameters which may not
be necessary to modify in most cases:

.. code-block:: yaml

  cluster:
    max_workers: 1
    method: null
  device: gpu
  microscope:
    beam:
      acceleration_voltage_spread: 8.0e-07
      defocus_drift: null
      drift: null
      electrons_per_angstrom: 30
      energy: 300
      energy_spread: 2.66e-06
      phi: 0
      source_spread: 0.1
      theta: 0
    detector:
      dqe: false
      nx: 1000
      ny: 1000
      origin:
      - 0
      - 0
      pixel_size: 1
    lens:
      c_10: -20000
      c_30: 2.7
      c_c: 2.7
      current_spread: 3.3e-07
    model: null
    phase_plate: false
  sample:
    box:
    - 1000
    - 1000
    - 1000
    centre:
    - 500
    - 500
    - 500
    ice: null
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
    sputter: null
  scan:
    axis:
    - 0
    - 1
    - 0
    exposure_time: 1
    mode: still
    num_images: 1
    start_angle: 0
    start_pos: 0
    step_angle: 0
    step_pos: 0
  simulation:
    division_thickness: 100
    ice: false
    inelastic_model: null
    margin: 100
    mp_loss_position: peak
    mp_loss_width: null
    padding: 100
    radiation_damage_model: false
    sensitivity_coefficient: 0.022
    slice_thickness: 3.0

Specifying molecule positions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following snippet will load one locally defined PDB file and will add a
single instance to the sample model. This will put the molecule in the centre
of the sample volume.

.. code-block:: yaml

  sample:
    molecules:
      local:
        - filename: myfile.pdb
          instances: 1

The following snippet will load one locally defined PDB file and will add a 10
instances to the sample model. This will give the molecules randomly assigned
positions and orientations within the sample volume.

.. code-block:: yaml

  sample:
    molecules:
      local:
        - filename: myfile.pdb
          instances: 10

The following snippet will load two locally defined PDB files and one model
from the PDB. The first model had two instances, the first of which has a
random position and random orientation. The second instance has defined
position and random orientation. The second molecule has two instances, the
first of which has random position and defined orientation and the second
instance has defined position and orientation. The PDB model has 10 instances
with random position and orientation.

.. code-block:: yaml

  sample:
    molecules:
      local:
        - filename: myfile.pdb
          instances: 
            - position: null
              orientation: null
            - position: [1, 2, 3]
              orientation: null
        - filename: another.pdb
          instances:
            - position: null
              orientation: [1, 2, 3]
            - position: [1, 2, 3]
              orientation: [1, 2, 3]
      pdb:
        - id: 4V5D
          instances: 10

Applying radiation damage
^^^^^^^^^^^^^^^^^^^^^^^^^

Parakeet implements a simple radiation damage model which uses an isotropic B
factor to blur the atomic potential during simulation. The B factor increases
linearly with the incident electron dose according to a sensitivity
coefficient. To apply the beam damage model you can set the following
parameters which will enable the beam damage model and simulate the images
using a dose symmetric scheme.

.. code-block:: yaml

  simulation:
    radiation_damage_model: true
    sensitivity_coefficient: 0.022

  scan:
    mode: dose_symmetric
