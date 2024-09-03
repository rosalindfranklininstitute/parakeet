import mrcfile
import numpy as np
import parakeet.beam
import os


def test_beam():
    beam = parakeet.beam.new(
        parakeet.config.Beam(
            energy=300,
            energy_spread=1,
            acceleration_voltage_spread=2,
            illumination_semiangle=0.1,
            electrons_per_angstrom=30,
        )
    )

    assert beam.energy == 300
    assert beam.energy_spread == 1
    assert beam.acceleration_voltage_spread == 2
    assert beam.illumination_semiangle == 0.1
    assert beam.electrons_per_angstrom == 30


def test_beam_incident_incident_wave(tmpdir):
    filename = os.path.join(tmpdir, "wave.mrc")

    h = mrcfile.new(filename)
    h.set_data(np.zeros((128, 128), dtype="float32"))
    del h

    beam = parakeet.beam.new(
        parakeet.config.Beam(
            energy=300,
            incident_wave=filename,
        )
    )

    assert beam.incident_wave.shape == (128, 128)
