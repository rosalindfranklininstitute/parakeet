import pytest
from math import exp
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
