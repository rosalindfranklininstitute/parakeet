import numpy as np
import pytest
import parakeet.scan


def test_still():
    scan = parakeet.scan.new(mode="still")
    assert np.all(np.equal(scan.axis, np.array([0, 1, 0])))
    assert np.all(np.equal(scan.angles, np.array([0])))
    assert np.all(np.equal(scan.positions, np.array([(0, 0, 0)])))


def test_tilt_series():
    scan = parakeet.scan.new(
        mode="tilt_series", axis=(1, 2, 3), start_angle=0, num_images=8, step_angle=45
    )
    assert np.all(np.equal(scan.axis, np.array([1, 2, 3])))
    assert np.all(np.equal(scan.angles, np.array([0, 45, 90, 135, 180, 225, 270, 315])))
    assert np.all(np.equal(scan.positions, np.array([(0, 0, 0) for i in range(8)])))


def test_helical_scan():
    scan = parakeet.scan.new(
        mode="helical_scan",
        axis=(0, 1, 0),
        start_angle=0,
        num_images=8,
        step_angle=45,
        start_pos=0,
        step_pos=10,
    )
    assert np.all(np.equal(scan.axis, np.array([0, 1, 0])))
    assert np.all(np.equal(scan.angles, np.array([0, 45, 90, 135, 180, 225, 270, 315])))
    assert np.all(
        np.equal(scan.positions, np.array([(0, i, 0) for i in range(0, 80, 10)]))
    )


def test_unknown():

    with pytest.raises(RuntimeError):
        scan = parakeet.scan.new("unknown")
