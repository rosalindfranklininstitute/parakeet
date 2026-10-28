Python API
==========

.. warning:: This is under developement

The functionality of the command line programs can be called from python as shown below.

Image IO functions
------------------

.. autofunction:: parakeet.io.new

.. autofunction:: parakeet.io.open

PDB file functions
------------------

.. autofunction:: parakeet.data.get_pdb

Configuration functions
-----------------------

.. autofunction:: parakeet.config.new

Sample functions
----------------

.. autofunction:: parakeet.sample.new

.. autofunction:: parakeet.sample.add_molecules

.. autofunction:: parakeet.sample.mill

.. autofunction:: parakeet.sample.sputter

Image simulation functions
--------------------------

.. autofunction:: parakeet.simulate.potential

.. autofunction:: parakeet.simulate.exit_wave

.. autofunction:: parakeet.simulate.optics

.. autofunction:: parakeet.simulate.image

.. autofunction:: parakeet.simulate.ctf

Analysis functions
------------------

.. autofunction:: parakeet.analyse.correct

.. autofunction:: parakeet.analyse.reconstruct

.. autofunction:: parakeet.analyse.average_particles

.. autofunction:: parakeet.analyse.average_all_particles

.. autofunction:: parakeet.analyse.extract

.. autofunction:: parakeet.analyse.refine

.. autofunction:: parakeet.run


Data models
-----------

.. autoclass:: parakeet.beam.Beam
  :members:
  
  .. automethod:: __init__

.. autoclass:: parakeet.detector.Detector
  :members:
  
  .. automethod:: __init__

.. autoclass:: parakeet.lens.Lens
  :members:
  
  .. automethod:: __init__

.. autoclass:: parakeet.microscope.Microscope
  :members:
  
  .. automethod:: __init__

.. autoclass:: parakeet.scan.Scan
  :members:
  
  .. automethod:: __init__

.. autoclass:: parakeet.sample.Sample
  :members:
  
  .. automethod:: __init__
