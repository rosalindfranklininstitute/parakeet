import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import starfile


class PoseSet:
    """
    A description of poses of the system being simulated.

    """

    def __init__(self, orientations: np.ndarray, shifts: np.ndarray):
        """
        Init the class

        Args:
            orientations (np.ndarray): (n, 3, 3) array of rotation matrices describing the pose of the system.
            shifts (np.ndarray): (n, 3) array of how to shift the sample volume (units: A)

        """
        self.orientations = R.from_matrix(orientations)
        self.shifts = shifts

    @property
    def euler_angles(self):
        """
        Euler angle representation of the orientations.

        The Euler angles are intrinsic, right handed rotations around ZYZ.
        This matches the convention used by XMIPP/RELION.

        """
        return self.orientations.as_euler(seq="ZYZ", degrees=True)

    @property
    def axis_angle(self):
        """
        Axis-angle representation of the orientations.

        Magnitude of the rotation vector corresponds to the angle in radians.

        """
        return self.orientations.as_rotvec()

    def __len__(self):
        """
        The number of poses

        """
        n_orientations = len(self.orientations)  # .shape[0]
        n_shifts = self.shifts.shape[0]
        if n_orientations != n_shifts:
            raise RuntimeError(
                "Number of orientations is different to the number of shifts."
            )
        return n_shifts

    def write_star_file(self, output_file):
        """Write a RELION format STAR file containing simulation metadata."""
        relion_eulers = self.orientations.inv().as_euler("ZYZ", degrees=True)
        relion_shifts = -self.shifts
        data = {
            "rlnAngleRot": relion_eulers[:, 0],
            "rlnAngleTilt": relion_eulers[:, 1],
            "rlnAnglePsi": relion_eulers[:, 2],
            "rlnOriginX": relion_shifts[:, 0],
            "rlnOriginY": relion_shifts[:, 1],
            "rlnOriginZ": relion_shifts[:, 2],
        }
        df = pd.DataFrame.from_dict(data)
        starfile.write(df, output_file, overwrite=True)
