#!/bin/bash

commands='
parakeet.analyse.average_all_particles
parakeet.analyse.average_particles
parakeet.analyse.correct
parakeet.analyse.extract_particles
parakeet.analyse.reconstruct
parakeet.analyse.refine
parakeet.config.edit
parakeet.config.show
parakeet.export
parakeet.read_pdb
parakeet.sample.add_molecules
parakeet.sample.mill
parakeet.sample.new
parakeet.sample.show
parakeet.sample.sputter
parakeet.simulate.ctf
parakeet.simulate.exit_wave
parakeet.simulate.image
parakeet.simulate.optics
parakeet.simulate.projected_potential
parakeet.simulate.simple
'

for command in ${commands}
do
  a=${command#parakeet.}
  b=${a//[.,_]/-}
  alias parakeet.${a}=parakeet.${b}
done
