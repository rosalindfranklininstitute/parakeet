#
# parakeet.sample.distribute.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy as np
from collections.abc import Iterable


class CuboidVolume(object):
    """
    Cuboid volume class

    """

    def __init__(self, lower: tuple, upper: tuple):
        """
        Initialise the volume

        Args:
            lower: The lower corner of the cuboid
            upper: The upper corner of the cuboid


        """

        assert len(lower) == len(upper)
        assert all([u > l for l, u in zip(lower, upper)])
        self.lower = lower
        self.upper = upper

    def generate_points(self, n: int) -> np.ndarray:
        """
        Generate the initial points

        Args:
            n: The number of points

        Returns:
            The initial coordinates

        """
        return np.random.uniform(self.lower, self.upper, size=(n, len(self.lower)))

    def reflect(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: np.ndarray,
        box_elasticity: float,
    ) -> tuple:
        """
        If points are outside the volume, reflect their positions and velocities

        Args:

            position: The positions of the particles
            velocity: The velocity of the particles
            radius: The radii of the particles
            box_elasticity: The rebound elasticity

        Returns:
            A tuple of the positions and velocities

        """
        radius = radius.reshape((-1, 1))
        for i in range(len(self.lower)):
            x = position[:, i : i + 1]
            V = velocity[:, i : i + 1]
            s1 = x > self.upper[i] - radius
            s2 = x < self.lower[i] + radius
            V[s1] = -V[s1] * box_elasticity
            V[s2] = -V[s2] * box_elasticity
            x[s1] = self.upper[i] - radius[s1]
            x[s2] = self.lower[i] + radius[s2]
            position[:, i : i + 1] = x
            velocity[:, i : i + 1] = V
        return position, velocity


class CylindricalVolume(object):
    """
    Cylinder volume class

    """

    def __init__(self, lower: float, upper: float, centre: list, radius: list):
        """
        Initialise the volume

        Args:
            lower: The lower corner of the cuboid
            upper: The upper corner of the cuboid
            centre: The list of centres along the length of the cylinder
            radius: The list of radius along the length of the cylinder

        """

        # Check the input
        assert upper > lower
        assert len(centre) == len(radius)
        assert len(centre) > 0

        # Add an extra position
        if len(radius) == 1:
            radius.append(radius[0])
            centre.append(centre[0])

        self.lower = lower
        self.upper = upper
        self.centre = centre
        self.radius = radius

        # Generate y coords to interpolate
        self.y = self.lower + np.arange(len(radius)) * (self.upper - self.lower) / (
            len(radius) - 1
        )

    def generate_points(self, n):
        """
        Generate the initial points

        Args:
            n: The number of points

        Returns:
            The initial coordinates

        """

        # Generate weights for y
        yx = np.interp(
            np.arange(0, len(self.y) - 1 + 0.01, 0.01),
            np.arange(0, len(self.y)),
            self.y,
        )
        px = np.interp(yx, self.y, np.array(self.radius) ** 2)
        px = px / np.sum(px)

        # Generate the y coords
        y = np.random.choice(yx, size=n, p=px)

        # Get the interpolated radius and centre
        rc = np.interp(y, self.y, self.radius)
        xc = np.interp(y, self.y, tuple(zip(*self.centre))[0])
        zc = np.interp(y, self.y, tuple(zip(*self.centre))[1])

        # Generate the coords
        t = np.random.uniform(0, 2 * np.pi, size=n)
        r = np.sqrt(np.random.uniform(0, 1, size=n)) * rc
        x = xc + r * np.cos(t)
        z = zc + r * np.sin(t)
        x.shape = (len(x), 1)
        y.shape = (len(y), 1)
        z.shape = (len(z), 1)
        return np.hstack((x, y, z))

    def reflect(self, position, velocity, radius, box_elasticity):
        """
        If points are outside the volume, reflect their positions and velocities

        Args:

            position: The positions of the particles
            velocity: The velocity of the particles
            radius: The radii of the particles
            box_elasticity: The rebound elasticity

        Returns:
            A tuple of the positions and velocities

        """

        # Get the components
        radius = radius.reshape((-1, 1))
        x = position[:, 0:1]
        y = position[:, 1:2]
        z = position[:, 2:3]
        Vx = velocity[:, 0:1]
        Vy = velocity[:, 1:2]
        Vz = velocity[:, 2:3]

        # Get the interpolated radius and centre
        rc = np.interp(y, self.y, self.radius)
        xc = np.interp(y, self.y, tuple(zip(*self.centre))[0])
        zc = np.interp(y, self.y, tuple(zip(*self.centre))[1])

        # Compute the x/z reflection
        r = np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
        t = np.arctan2(z - zc, x - xc)
        Vr = np.sqrt(Vx**2 + Vz**2)
        Vt = np.arctan2(Vz, Vx)
        s = r > (rc - radius)
        Vr[s] -= Vr[s] * box_elasticity
        r[s] = rc[s] - radius[s]
        x[s] = xc[s] + r[s] * np.cos(t[s])
        z[s] = zc[s] + r[s] * np.sin(t[s])
        Vx[s] = Vr[s] * np.cos(Vt[s])
        Vz[s] = Vr[s] * np.sin(Vt[s])

        # Compute the y reflection
        s1 = y > self.upper - radius
        s2 = y < self.lower + radius
        Vy[s1] = -Vy[s1] * box_elasticity
        Vy[s2] = -Vy[s2] * box_elasticity
        y[s1] = self.upper - radius[s1]
        y[s2] = self.lower + radius[s2]

        # Return the coords
        return np.hstack((x, y, z)), np.hstack((Vx, Vy, Vz))


def shape_volume_object(centre: tuple, shape: dict):
    """
    Make a shape volume object

    Args:
        centre: The centre of the volume
        shape: The shape description

    Returns:
        The volume object

    """

    def make_cube_volume(centre, cube, margin):
        length = cube["length"]
        lower = np.array(centre) - length / 2.0
        upper = lower + length
        lower += np.array(margin)
        upper -= np.array(margin)
        return CuboidVolume(lower, upper)

    def make_cuboid_volume(centre, cuboid, margin):
        length_x = cuboid["length_x"]
        length_y = cuboid["length_y"]
        length_z = cuboid["length_z"]
        length = np.array((length_x, length_y, length_z))
        lower = np.array(centre) - length / 2.0
        upper = lower + length
        lower += np.array(margin)
        upper -= np.array(margin)
        return CuboidVolume(lower, upper)

    def make_cylinder_volume(centre, cylinder, margin):
        # Get the cylinder params
        length = cylinder["length"]
        radius = cylinder["radius"]
        offset_x = cylinder.get("offset_x", None)
        offset_z = cylinder.get("offset_z", None)
        axis = cylinder.get("axis", (0, 1, 0))
        assert np.all(np.equal(axis, (0, 1, 0)))

        # Make into a list for radius and offset
        if not isinstance(radius, Iterable):
            radius = [radius]
        if offset_x is None:
            offset_x = [0] * len(radius)
        if offset_z is None:
            offset_z = [0] * len(radius)

        # Get upper lower and centre
        lower = centre[1] - length / 2.0
        upper = lower + length
        centre = list(
            np.array((centre[0], centre[2])) + np.array(list(zip(offset_x, offset_z)))
        )

        # Add a margin
        lower += margin[1]
        upper -= margin[1]
        radius = [max(1, r - margin[0]) for r in radius]

        # Return volume
        return CylindricalVolume(lower, upper, centre, radius)

    return {
        "cube": make_cube_volume,
        "cuboid": make_cuboid_volume,
        "cylinder": make_cylinder_volume,
    }[shape["type"]](centre, shape[shape["type"]], shape["margin"])


def distribute_particles_uniformly(
    volume, radius: np.ndarray, max_iterations: int = 1000
) -> np.ndarray:
    """
    Find n random non overlapping positions for cuboids within a volume

    Args:
        volume: The volume object
        radius: The list of bounding sphere radii
        max_iterations: The maximum number of iterations

    Returns:
        list: A list of centre positions

    """

    def update(volume, position, radius, max_iterations):
        assert len(position) == len(radius)

        # Get the initial velocities
        size = 3 * np.std(position, axis=0)
        velocity = 0.01 * np.random.uniform(-size / 2, size / 2, size=position.shape)

        # Set the box elasticity
        box_elasticity = 1

        # Get the min separation between all position
        separation = radius[np.newaxis, :] + radius[:, np.newaxis]

        # Create a mask to exclude diagonals
        diagonal_mask = np.ones(separation.shape, dtype=bool)
        np.fill_diagonal(diagonal_mask, 0)

        # Set the resistance
        resistance = 0

        # Avoid division by zero
        epsilon = 1e-6

        # Set the time step
        dt = 0.1

        # Loop through the iterations
        for t in range(max_iterations):
            # Update the current position
            position += velocity * dt

            # Add some resistance to damp the velocity
            velocity -= velocity * resistance

            # Reflect particle if outside box
            position, velocity = volume.reflect(
                position, velocity, radius, box_elasticity
            )

            # Compute the distance between particles
            dr2 = np.sum(
                (position[np.newaxis, :, :] - position[:, np.newaxis, :]) ** 2, axis=2
            )

            # Find number of overlaps
            s = dr2 <= (separation * 1.01) ** 2
            i_list, j_list = s.nonzero()
            s = j_list > i_list
            i_list = i_list[s]
            j_list = j_list[s]
            print("Step: %d/%d; # overlaps: %d" % (t + 1, max_iterations, len(i_list)))

            # Loop through the overlaps and calculate an elastic collision
            for i, j in zip(i_list, j_list):
                # Compute the distance between the position and min separation
                dp = position[i] - position[j]
                dr2 = np.sum(dp**2) + epsilon**2
                dr = np.sqrt(dr2)
                d = (radius[i] + radius[j]) * 1.01

                # Update the velocity of the particles
                Vn = np.dot(velocity[i] - velocity[j], dp) * dp / dr2
                velocity[i] -= Vn
                velocity[j] += Vn

                # Update the position of the particles
                Pn = (d - dr) * (dp / dr) * 0.5
                position[i] += Pn
                position[j] -= Pn

            # Compute the current distance
            dr2 = np.sum(
                (position[np.newaxis, :, :] - position[:, np.newaxis, :]) ** 2, axis=2
            )

            # If all distances are greater than the min separation then break
            if np.all(dr2[diagonal_mask] > separation[diagonal_mask] ** 2):
                break

        # Check the minimum particle separation
        if np.any(dr2[diagonal_mask] < separation[diagonal_mask] ** 2):
            raise RuntimeError("Unable to place %d particles" % len(position))
        else:
            print("Generated positions after step: %d/%d" % (t + 1, max_iterations))

        # Return the position
        return position

    # The number of particles
    num_particles = len(radius)

    # Generate the initial positions
    position = volume.generate_points(num_particles)

    # Update the coordinates
    return update(volume, position, radius, max_iterations)
