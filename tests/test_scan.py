import numpy as np
import pytest
import parakeet.scan


def test_none():
    scan = parakeet.scan.new(mode=None)
    assert np.all(np.equal(scan.axes, np.array([[0, 1, 0]])))
    assert np.all(np.equal(scan.angles, np.array([0])))
    assert np.all(np.equal(scan.position, np.array([(0, 0, 0)])))


def test_manual():
    scan = parakeet.scan.new(
        mode="manual",
        axis=(0, 1, 0),
        angles=[1, 2, 3],
        positions=[4, 5, 6],
        defocus_offset=[0, 500, 1000],
    )
    assert np.all(np.equal(scan.axes, np.array([[0, 1, 0]])))
    assert np.allclose(scan.angles, np.array([1, 2, 3]))
    assert np.allclose(scan.position, np.array([(0, i, 0) for i in [4, 5, 6]]))


def test_still():
    scan = parakeet.scan.new(mode="still")
    assert np.all(np.equal(scan.axes, np.array([[0, 1, 0]])))
    assert np.all(np.equal(scan.angles, np.array([0])))
    assert np.all(np.equal(scan.position, np.array([(0, 0, 0)])))


def test_tilt_series():
    scan = parakeet.scan.new(
        mode="tilt_series", axis=(1, 2, 3), start_angle=1, num_images=8, step_angle=45
    )
    axis = np.array((1, 2, 3)) / np.linalg.norm((1, 2, 3))
    assert np.allclose(scan.axes, np.array([axis] * 8))
    assert np.allclose(scan.angles, 1 + np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    assert np.allclose(scan.position, np.array([(0, 0, 0) for i in range(8)]))


def test_dose_symmetric():
    scan = parakeet.scan.new(
        mode="dose_symmetric",
        axis=(1, 2, 3),
        start_angle=-8.5,
        num_images=8,
        step_angle=2,
    )
    axis = np.array((1, 2, 3)) / np.linalg.norm((1, 2, 3))
    assert np.allclose(scan.axes, np.array([axis] * 8))
    assert np.allclose(
        scan.angles, np.array([-0.5, 1.5, -2.5, 3.5, -4.5, 5.5, -6.5, -8.5])
    )
    assert np.allclose(scan.position, np.array([(0, 0, 0) for i in range(8)]))


def test_helical_scan():
    scan = parakeet.scan.new(
        mode="helical_scan",
        axis=(0, 1, 0),
        start_angle=1,
        num_images=8,
        step_angle=45,
        start_pos=0,
        step_pos=10,
    )

    assert np.allclose(scan.axes, np.array([[0, 1, 0]] * 8))
    assert np.allclose(scan.angles, 1 + np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    assert np.allclose(scan.position, np.array([(0, i, 0) for i in range(0, 80, 10)]))


def test_nhelix():
    scan = parakeet.scan.new(
        mode="nhelix",
        axis=(0, 1, 0),
        start_angle=1,
        num_images=8,
        num_nhelix=2,
        step_angle=10,
        start_pos=0,
        step_pos=10,
    )
    angles = 1 + np.concatenate(
        [[0, 10, 20, 30, 40, 50, 60, 70], [5, 15, 25, 35, 45, 55, 65, 75]]
    )
    positions = np.concatenate(
        [[(0, 0, 0) for i in range(0, 80, 10)], [(0, 10, 0) for i in range(0, 80, 10)]]
    )

    assert np.allclose(scan.axes, np.array([[0, 1, 0]] * 8 * 2))
    assert np.allclose(scan.angles, angles)
    assert np.allclose(scan.position, positions)


def test_single_particle():
    scan = parakeet.scan.new(
        mode="single_particle",
        num_images=8,
        defocus_offset=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
    )
    assert len(scan) == 8


def test_grid_scan():
    scan = parakeet.scan.new(
        mode="grid_scan",
        axis=(0, 1, 0),
        angles=[1, 2, 3, 4],
        start_pos=(0, 0),
        step_pos=(10, 10),
        num_images=(10, 10),
    )
    angles = np.array([[1, 2, 3, 4]] * 10 * 10).T.flatten()
    positions = np.array(
        [(x * 10, y * 10, 0) for a in range(4) for y in range(10) for x in range(10)]
    )
    axis = np.array([[0, 1, 0]] * 10 * 10 * 4)
    assert np.allclose(scan.axes, axis)
    assert np.allclose(scan.angles, angles)
    assert np.allclose(scan.position, positions)


def test_beam_tilt():
    scan = parakeet.scan.new(
        mode="beam_tilt",
        axis=(0, 1, 0),
        angles=[1, 2, 3, 4],
        positions=[10, 20, 30, 40],
        theta=[0, 0, 0, 0, 0, 0],
        phi=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    angles = np.array([[1, 2, 3, 4]] * 6).T.flatten()
    positions = np.array([[10 * i for i in range(1, 5)]] * 6).T.flatten()
    positions = np.array([np.array((0, 1, 0)) * p for p in positions])
    theta = np.array([[0, 0, 0, 0, 0, 0]] * 4).flatten()
    phi = np.array([[0, 0.1, 0.2, 0.3, 0.4, 0.5]] * 4).flatten()
    axis = np.array([[0, 1, 0]] * 4 * 6)
    assert np.allclose(scan.axes, axis)
    assert np.allclose(scan.angles, angles)
    assert np.allclose(scan.position, positions)
    assert np.allclose(scan.beam_tilt_theta, theta)
    assert np.allclose(scan.beam_tilt_phi, phi)


def test_unknown():
    with pytest.raises(KeyError):
        scan = parakeet.scan.new("unknown")
