import parakeet.beam


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
