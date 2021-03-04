import numpy
import pytest
import amplus.scan


def test_still():
    scan = amplus.scan.new(mode="still")
    assert scan.axis == (0, 1, 0)
    assert scan.angles == [0]
    assert scan.positions == [0]


def test_tilt_series():
    scan = amplus.scan.new(
        mode="tilt_series", axis=(1, 2, 3), start_angle=0, num_images=8, step_angle=45
    )
    assert scan.axis == (1, 2, 3)
    assert numpy.all(numpy.equal(scan.angles, [0, 45, 90, 135, 180, 225, 270, 315]))
    assert numpy.all(numpy.equal(scan.positions, [0, 0, 0, 0, 0, 0, 0, 0]))


def test_helical_scan():
    scan = amplus.scan.new(
        mode="helical_scan",
        axis=(1, 2, 3),
        start_angle=0,
        num_images=8,
        step_angle=45,
        start_pos=0,
        step_pos=10,
    )
    assert scan.axis == (1, 2, 3)
    assert numpy.all(numpy.equal(scan.angles, [0, 45, 90, 135, 180, 225, 270, 315]))
    assert numpy.all(numpy.equal(scan.positions, [0, 10, 20, 30, 40, 50, 60, 70]))


def test_unknown():

    with pytest.raises(RuntimeError):
        scan = amplus.scan.new("unknown")
