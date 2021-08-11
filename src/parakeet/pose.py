from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class PoseSet:
    """A description of poses of the system being simulated.

    Parameters
    __________

    orientations : np.ndarray
        (n, 3, 3) array of rotation matrices describing the pose of the system.
    shifts : np.ndarray
        (n, 3) array of how to shift the sample volume (units: A)
    """
    orientations: np.ndarray
    shifts: np.ndarray

    def __post_init__(self):
        self._orientations = self.orientations
        self.orientations = R.from_matrix(self._orientations)

    @property
    def euler_angles(self):
        """Euler angle representation of the orientations.

        The Euler angles are intrinsic, right handed rotations around ZYZ.
        This matches the convention used by XMIPP/RELION.
        """
        return self.orientations.as_euler(seq='ZYZ', degrees=True)

    @property
    def axis_angle(self):
        """Axis-angle representation of the orientations.

        Magnitude of the rotation vector corresponds to the angle in radians.
        """
        return self.orientations.as_rotvec()

    def __len__(self):
        n_orientations = self.orientations.shape[0]
        n_shifts = self.shifts.shape[0]
        if n_orientations != n_shifts:
            raise RuntimeError(
                'Number of orientations is different to the number of shifts.'
            )
        return n_shifts