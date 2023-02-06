import numpy as np
from parakeet.sample.distribute import CuboidVolume
from parakeet.sample.distribute import CylindricalVolume
from parakeet.sample.distribute import distribute_particles_uniformly


def test_cube():
    radius = np.random.uniform(100, 250, size=1000)

    lower = (0, 0, 0)
    upper = (10000, 10000, 10000)
    volume = CuboidVolume(lower, upper)

    points = distribute_particles_uniformly(volume, radius)

    assert np.all(points >= lower)
    assert np.all(points <= upper)
    for i in range(len(radius) - 1):
        for j in range(i + 1, len(radius)):
            p1 = points[i]
            p2 = points[j]
            r1 = radius[i]
            r2 = radius[j]
            assert np.sqrt(np.sum((p1 - p2) ** 2)) > r1 + r2


def test_cuboid():
    radius = np.random.uniform(100, 250, size=1000)

    lower = (0, 0, 0)
    upper = (10000, 1000, 10000)
    volume = CuboidVolume(lower, upper)

    points = distribute_particles_uniformly(volume, radius)

    assert np.all(points >= lower)
    assert np.all(points <= upper)
    for i in range(len(radius) - 1):
        for j in range(i + 1, len(radius)):
            p1 = points[i]
            p2 = points[j]
            r1 = radius[i]
            r2 = radius[j]
            assert np.sqrt(np.sum((p1 - p2) ** 2)) > r1 + r2


def test_cylinder_1():
    radius = np.random.uniform(100, 250, size=1000)

    lower = 0
    upper = 10000
    volume = CylindricalVolume(lower, upper, [(0, 0)], [1500])

    points = distribute_particles_uniformly(volume, radius)

    x = points[:, 0:1]
    y = points[:, 1:2]
    z = points[:, 2:3]
    rc = np.interp(y, volume.y, volume.radius)
    xc = np.interp(y, volume.y, tuple(zip(*volume.centre))[0])
    zc = np.interp(y, volume.y, tuple(zip(*volume.centre))[1])
    r = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    t = np.arctan2(z - zc, x - xc)
    assert np.all(r <= rc)
    assert np.all(points[:, 1] >= lower)
    assert np.all(points[:, 1] <= upper)
    for i in range(len(radius) - 1):
        for j in range(i + 1, len(radius)):
            p1 = points[i]
            p2 = points[j]
            r1 = radius[i]
            r2 = radius[j]
            assert np.sqrt(np.sum((p1 - p2) ** 2)) > r1 + r2


def test_cylinder_2():
    radius = np.random.uniform(100, 250, size=400)

    lower = 0
    upper = 10000
    volume = CylindricalVolume(lower, upper, [(0, 0), (0, 0)], [1500, 500])

    points = distribute_particles_uniformly(volume, radius)

    x = points[:, 0:1]
    y = points[:, 1:2]
    z = points[:, 2:3]
    rc = np.interp(y, volume.y, volume.radius)
    xc = np.interp(y, volume.y, tuple(zip(*volume.centre))[0])
    zc = np.interp(y, volume.y, tuple(zip(*volume.centre))[1])
    r = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    t = np.arctan2(z - zc, x - xc)
    assert np.all(r <= rc)
    assert np.all(points[:, 1] >= lower)
    assert np.all(points[:, 1] <= upper)
    for i in range(len(radius) - 1):
        for j in range(i + 1, len(radius)):
            p1 = points[i]
            p2 = points[j]
            r1 = radius[i]
            r2 = radius[j]
            assert np.sqrt(np.sum((p1 - p2) ** 2)) > r1 + r2


def test_cylinder_3():
    radius = np.random.uniform(100, 250, size=900)

    lower = 0
    upper = 10000
    volume = CylindricalVolume(
        lower,
        upper,
        [
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ],
        [1500, 1450, 1400, 1350, 1300, 1250, 1200, 1150, 1100, 1050],
    )

    points = distribute_particles_uniformly(volume, radius)

    x = points[:, 0:1]
    y = points[:, 1:2]
    z = points[:, 2:3]
    rc = np.interp(y, volume.y, volume.radius)
    xc = np.interp(y, volume.y, tuple(zip(*volume.centre))[0])
    zc = np.interp(y, volume.y, tuple(zip(*volume.centre))[1])
    r = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    t = np.arctan2(z - zc, x - xc)
    assert np.all(r <= rc)
    assert np.all(points[:, 1] >= lower)
    assert np.all(points[:, 1] <= upper)
    for i in range(len(radius) - 1):
        for j in range(i + 1, len(radius)):
            p1 = points[i]
            p2 = points[j]
            r1 = radius[i]
            r2 = radius[j]
            assert np.sqrt(np.sum((p1 - p2) ** 2)) > r1 + r2


def test_cylinder_4():
    radius = np.random.uniform(100, 250, size=900)

    lower = 0
    upper = 10000
    volume = CylindricalVolume(
        lower,
        upper,
        [
            (0, 0),
            (100, -100),
            (200, -200),
            (300, -300),
            (200, -200),
            (100, -100),
            (0, 0),
            (-100, 100),
            (-200, 200),
            (-300, 300),
        ],
        [1500, 1450, 1400, 1350, 1300, 1250, 1200, 1150, 1100, 1050],
    )

    points = distribute_particles_uniformly(volume, radius)

    x = points[:, 0:1]
    y = points[:, 1:2]
    z = points[:, 2:3]
    rc = np.interp(y, volume.y, volume.radius)
    xc = np.interp(y, volume.y, tuple(zip(*volume.centre))[0])
    zc = np.interp(y, volume.y, tuple(zip(*volume.centre))[1])
    r = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    t = np.arctan2(z - zc, x - xc)
    assert np.all(r <= rc)
    assert np.all(points[:, 1] >= lower)
    assert np.all(points[:, 1] <= upper)
    for i in range(len(radius) - 1):
        for j in range(i + 1, len(radius)):
            p1 = points[i]
            p2 = points[j]
            r1 = radius[i]
            r2 = radius[j]
            assert np.sqrt(np.sum((p1 - p2) ** 2)) > r1 + r2
