import numpy
import pytest
import elfantasma.scan


def test_still():
    scan = elfantasma.scan.new(mode="still")
    assert scan.axis == (1, 0, 0)
    assert scan.angles == [0]
    assert scan.positions == [0]


def test_tilt_series():
    scan = elfantasma.scan.new(
        mode="tilt_series", axis=(1, 2, 3), start_angle=0, stop_angle=360, step_angle=45
    )
    assert scan.axis == (1, 2, 3)
    assert numpy.all(numpy.equal(scan.angles, [0, 45, 90, 135, 180, 225, 270, 315]))
    assert numpy.all(numpy.equal(scan.positions, [0, 0, 0, 0, 0, 0, 0, 0]))


def test_helical_scan():
    scan = elfantasma.scan.new(
        mode="helical_scan",
        axis=(1, 2, 3),
        start_angle=0,
        stop_angle=360,
        step_angle=45,
        start_pos=0,
        stop_pos=80,
    )
    assert scan.axis == (1, 2, 3)
    assert numpy.all(numpy.equal(scan.angles, [0, 45, 90, 135, 180, 225, 270, 315]))
    assert numpy.all(numpy.equal(scan.positions, [0, 10, 20, 30, 40, 50, 60, 70]))


def test_unknown():

    with pytest.raises(RuntimeError):
        scan = elfantasma.scan.new("unknown")
