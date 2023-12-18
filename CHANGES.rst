Parakeet 0.5.0 (2023-12-18)
===========================

Features
--------

- Better inelastic models merged (#32)
- Enable add_molecules to be called multiple times. If this is done then no guarantee is made regarding overlapping particles. New particles which overlap old ones will delete the atoms of the old particles. The positions and orientations of all particles will be recorded in the sample.h5 file. (#57)
- Added ability to set objective aperture cutoff frequency (#61)


Parakeet v0.4.6 (2023-10-20)
============================

Features
--------

- Implemented phase plate functionality (#39)
- Added the ability to easily perform a grid scan (#50)
- Added CBED program for CBED simulation (#51)
- Added ability to set GPU id (#56)


Improved Documentation
----------------------

- Improved installation documentation (#44)


Parakeet 0.4.4.dev29+g7b20723.d20230217 (2023-03-01)
====================================================

Bugfixes
--------

- Exit wave padding can now be set to zero without an error (#45)
