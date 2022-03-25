Python API
==========

High level API
--------------

.. warning:: This is under developement

The functionality of the command line programs can be called from python as shown below.

.. autofunction:: parakeet.config.new

.. autofunction:: parakeet.sample.new

.. autofunction:: parakeet.sample.add_molecules

.. autofunction:: parakeet.simulate.exit_wave

.. autofunction:: parakeet.simulate.optics

.. autofunction:: parakeet.simulate.image

.. autofunction:: parakeet.analyse.correct

.. autofunction:: parakeet.analyse.reconstruct

.. autofunction:: parakeet.analyse.average_particles

.. autofunction:: parakeet.analyse.extract

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
