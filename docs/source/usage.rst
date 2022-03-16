Command line programs
=====================

Config file manipulation programs
---------------------------------

parakeet.config.new
^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.config
   :func: get_new_parser
   :prog: parakeet.config.new

parakeet.config.show
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.config
   :func: get_show_parser
   :prog: parakeet.config.show

parakeet.config.edit
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.config
   :func: get_edit_parser
   :prog: parakeet.config.edit

Sample manipulation programs
----------------------------

parakeet.sample.new
^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample
   :func: get_new_parser
   :prog: parakeet.sample.new

parakeet.sample.add_molecules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample
   :func: get_add_molecules_parser
   :prog: parakeet.sample.add_molecules

parakeet.sample.mill
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample
   :func: get_mill_parser
   :prog: parakeet.sample.mill

parakeet.sample.sputter
^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample
   :func: get_sputter_parser
   :prog: parakeet.sample.sputter

parakeet.sample.show
^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.sample
   :func: get_show_parser
   :prog: parakeet.sample.show


Image Simulation programs
-------------------------

parakeet.simulate.projected_potential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate
   :func: get_projected_potential_parser
   :prog: parakeet.simulate.projected_potential

parakeet.simulate.exit_wave
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate
   :func: get_exit_wave_parser
   :prog: parakeet.simulate.exit_wave

parakeet.simulate.optics
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate
   :func: get_optics_parser
   :prog: parakeet.simulate.optics

parakeet.simulate.ctf
^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate
   :func: get_ctf_parser
   :prog: parakeet.simulate.ctf

parakeet.simulate.image
^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate
   :func: get_image_parser
   :prog: parakeet.simulate.image

parakeet.simulate.simple
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.simulate
   :func: get_simple_parser
   :prog: parakeet.simulate.simple


Analysis programs
-----------------

parakeet.analyse.reconstruct
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse
   :func: get_reconstruct_parser
   :prog: parakeet.analyse.reconstruct

parakeet.analyse.correct
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse
   :func: get_correct_parser
   :prog: parakeet.analyse.correct

parakeet.analyse.average_particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse
   :func: get_average_particles_parser
   :prog: parakeet.analyse.average_particles

parakeet.analyse.average_all_particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse
   :func: get_average_all_particles_parser
   :prog: parakeet.analyse.average_all_particles

parakeet.analyse.extract
^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse
   :func: get_extract_parser
   :prog: parakeet.analyse.extract


parakeet.analyse.refine
^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line.analyse
   :func: get_refine_parser
   :prog: parakeet.analyse.refine


Other programs
--------------

parakeet.export
^^^^^^^^^^^^^^^

.. argparse::
   :module: parakeet.command_line
   :func: get_export_parser
   :prog: parakeet.export
