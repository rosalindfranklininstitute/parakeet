import pytest
import numpy as np
from math import exp, sqrt
from parakeet import inelastic


def test_zero_loss_fraction():
    cube = {"type": "cube", "cube": {"length": 3150}}

    cuboid = {
        "type": "cuboid",
        "cuboid": {"length_x": 1000, "length_y": 1000, "length_z": 3150},
    }

    cylinder = {"type": "cylinder", "cylinder": {"length": 1000, "radius": 1575}}

    angle = 0
    fraction = inelastic.zero_loss_fraction(cube, angle)
    assert fraction == pytest.approx(exp(-1))

    angle = 60
    fraction = inelastic.zero_loss_fraction(cube, angle)
    assert fraction == pytest.approx(exp(-2))

    angle = 0
    fraction = inelastic.zero_loss_fraction(cuboid, angle)
    assert fraction == pytest.approx(exp(-1))

    angle = 60
    fraction = inelastic.zero_loss_fraction(cuboid, angle)
    assert fraction == pytest.approx(exp(-2))

    angle = 0
    fraction = inelastic.zero_loss_fraction(cylinder, angle)
    assert fraction == pytest.approx(exp(-1))

    angle = 60
    fraction = inelastic.zero_loss_fraction(cylinder, angle)
    assert fraction == pytest.approx(exp(-1))


def test_mp_loss_fraction():
    cube = {"type": "cube", "cube": {"length": 3150}}

    cuboid = {
        "type": "cuboid",
        "cuboid": {"length_x": 1000, "length_y": 1000, "length_z": 3150},
    }

    cylinder = {"type": "cylinder", "cylinder": {"length": 1000, "radius": 1575}}

    angle = 0
    fraction = inelastic.mp_loss_fraction(cube, angle)
    assert fraction == pytest.approx(1 - exp(-1))

    angle = 60
    fraction = inelastic.mp_loss_fraction(cube, angle)
    assert fraction == pytest.approx(1 - exp(-2))

    angle = 0
    fraction = inelastic.mp_loss_fraction(cuboid, angle)
    assert fraction == pytest.approx(1 - exp(-1))

    angle = 60
    fraction = inelastic.mp_loss_fraction(cuboid, angle)
    assert fraction == pytest.approx(1 - exp(-2))

    angle = 0
    fraction = inelastic.mp_loss_fraction(cylinder, angle)
    assert fraction == pytest.approx(1 - exp(-1))

    angle = 60
    fraction = inelastic.mp_loss_fraction(cylinder, angle)
    assert fraction == pytest.approx(1 - exp(-1))


def test_fraction_of_electrions():
    cube = {"type": "cube", "cube": {"length": 3150}}

    fraction = inelastic.fraction_of_electrons(cube, 0, "zero_loss")
    assert fraction == pytest.approx(exp(-1))
    fraction = inelastic.fraction_of_electrons(cube, 60, "zero_loss")
    assert fraction == pytest.approx(exp(-2))

    fraction = inelastic.fraction_of_electrons(cube, 0, "mp_loss")
    assert fraction == pytest.approx(1 - exp(-1))
    fraction = inelastic.fraction_of_electrons(cube, 60, "mp_loss")
    assert fraction == pytest.approx(1 - exp(-2))

    fraction = inelastic.fraction_of_electrons(cube, 0, "cc_corrected")
    assert fraction == pytest.approx(1.0)
    fraction = inelastic.fraction_of_electrons(cube, 60, "cc_corrected")
    assert fraction == pytest.approx(1.0)

    fraction = inelastic.fraction_of_electrons(cube, 0, "unfiltered")
    assert fraction == pytest.approx(1.0)
    fraction = inelastic.fraction_of_electrons(cube, 60, "unfiltered")
    assert fraction == pytest.approx(1.0)


def test_most_probable_loss():
    cube = {"type": "cube", "cube": {"length": 3150}}

    peak, sigma = inelastic.most_probable_loss(300, cube, 0)
    assert peak == pytest.approx(17.92806966151457)
    assert sigma == pytest.approx(5.300095984425282)


@pytest.mark.parametrize("thickness", [100, 1000, 4000])
def test_get_energy_bins(thickness):
    bin_energy, bin_spread, bin_weight = inelastic.get_energy_bins(
        energy=300000, thickness=thickness, energy_spread=0.798
    )

    assert np.min(bin_weight) >= 0
    assert np.max(bin_weight) <= 1
    assert np.isclose(np.sum(bin_weight), 1)
    assert np.argmax(bin_weight) == 2
    assert np.max(bin_spread) <= sqrt(5**2 / 12) * sqrt(2) + 0.05
    assert np.min(bin_spread) >= 0
