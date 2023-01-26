Command line programs
=====================

Config file manipulation programs
---------------------------------

parakeet.config.new
^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.config._new
   :func: get_parser
   :prog: parakeet.config.new

parakeet.config.show
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.config._show
   :func: get_parser
   :prog: parakeet.config.show

parakeet.config.edit
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.config._edit
   :func: get_parser
   :prog: parakeet.config.edit

Sample manipulation programs
----------------------------

parakeet.sample.new
^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample._new
   :func: get_parser
   :prog: parakeet.sample.new

parakeet.sample.add_molecules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample._add_molecules
   :func: get_parser
   :prog: parakeet.sample.add_molecules

parakeet.sample.mill
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample._mill
   :func: get_parser
   :prog: parakeet.sample.mill

parakeet.sample.sputter
^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample._sputter
   :func: get_parser
   :prog: parakeet.sample.sputter

parakeet.sample.show
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample._show
   :func: get_parser
   :prog: parakeet.sample.show


Image Simulation programs
-------------------------

parakeet.simulate.potential
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate._potential
   :func: get_parser
   :prog: parakeet.simulate.potential

parakeet.simulate.exit_wave
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate._exit_wave
   :func: get_parser
   :prog: parakeet.simulate.exit_wave

parakeet.simulate.optics
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate._optics
   :func: get_parser
   :prog: parakeet.simulate.optics

parakeet.simulate.ctf
^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate._ctf
   :func: get_parser
   :prog: parakeet.simulate.ctf

parakeet.simulate.image
^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate._image
   :func: get_parser
   :prog: parakeet.simulate.image

parakeet.simulate.simple
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate._simple
   :func: get_parser
   :prog: parakeet.simulate.simple


Analysis programs
-----------------

parakeet.analyse.reconstruct
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse._reconstruct
   :func: get_parser
   :prog: parakeet.analyse.reconstruct

parakeet.analyse.correct
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse._correct
   :func: get_parser
   :prog: parakeet.analyse.correct

parakeet.analyse.average_particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse._average_particles
   :func: get_parser
   :prog: parakeet.analyse.average_particles

parakeet.analyse.average_all_particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse._average_all_particles
   :func: get_parser
   :prog: parakeet.analyse.average_all_particles

parakeet.analyse.extract
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse._extract
   :func: get_parser
   :prog: parakeet.analyse.extract


parakeet.analyse.refine
^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse._refine
   :func: get_parser
   :prog: parakeet.analyse.refine


PDB file programs
------------------

parakeet.pdb.read
^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.pdb._read
   :func: get_parser
   :prog: parakeet.pdb.read

parakeet.pdb.get
^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.pdb._get
   :func: get_parser
   :prog: parakeet.pdb.get


Other programs
--------------

parakeet.export
^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line._export
   :func: get_parser
   :prog: parakeet.export

parakeet.run
^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line._run
   :func: get_parser
   :prog: parakeet.run

parakeet
^^^^^^^^

.. argparse::
   :module: parakeet.command_line._main
   :func: get_parser
   :prog: parakeet
