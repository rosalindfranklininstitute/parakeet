#
# parakeet.scan.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R


class Scan(object):
    """
    A scan of angles around a single rotation axis.

    """

    def __init__(
        self,
        image_number: np.ndarray = None,
        fraction_number: np.ndarray = None,
        axis: np.ndarray = None,
        angle: np.ndarray = None,
        shift: np.ndarray = None,
        shift_delta: np.ndarray = None,
        beam_tilt_theta: np.ndarray = None,
        beam_tilt_phi: np.ndarray = None,
        electrons_per_angstrom: np.ndarray = None,
        defocus_offset: np.ndarray = None,
        exposure_time: float = 1,
        is_uniform_angular_scan: bool = False,
    ):
        """
        Initialise the scan

        """
        self.is_uniform_angular_scan = is_uniform_angular_scan

        if image_number is None:
            image_number = np.array([0])

        if fraction_number is None:
            fraction_number = np.zeros(len(image_number))

        if axis is None:
            axis = np.zeros((len(image_number), 3))

        if angle is None:
            angle = np.zeros(len(image_number))

        if shift is None:
            shift = np.zeros((len(image_number), 3))

        if shift_delta is None:
            shift_delta = np.zeros((len(image_number), 3))

        if beam_tilt_theta is None:
            beam_tilt_theta = np.zeros(len(image_number))

        if beam_tilt_phi is None:
            beam_tilt_phi = np.zeros(len(image_number))

        if electrons_per_angstrom is None:
            electrons_per_angstrom = np.ones(len(image_number))

        if defocus_offset is None:
            defocus_offset = np.zeros(len(image_number))

        self.data = pd.DataFrame(
            data={
                "image_number": image_number,
                "fraction_number": fraction_number,
                "axis_x": axis[:, 0],
                "axis_y": axis[:, 1],
                "axis_z": axis[:, 2],
                "angle": np.radians(angle),
                "shift_x": shift[:, 0],
                "shift_y": shift[:, 1],
                "shift_z": shift[:, 2],
                "shift_delta_x": shift_delta[:, 0],
                "shift_delta_y": shift_delta[:, 1],
                "shift_delta_z": shift_delta[:, 2],
                "beam_tilt_theta": beam_tilt_theta,
                "beam_tilt_phi": beam_tilt_phi,
                "electrons_per_angstrom": electrons_per_angstrom,
                "defocus_offset": defocus_offset,
                "exposure_time": np.ones(len(axis)) * exposure_time,
            }
        )

    @property
    def image_number(self) -> np.ndarray:
        """
        Get the image number

        """
        return self.data["image_number"]

    @property
    def fraction_number(self) -> np.ndarray:
        """
        Get the movie fraction number

        """
        return self.data["fraction_number"]

    @property
    def orientation(self) -> np.ndarray:
        """
        Get the orientations

        """
        return self.axes * np.array(self.data["angle"])[:, np.newaxis]

    @property
    def shift(self) -> np.ndarray:
        """
        Get the shifts

        """
        return np.array(self.data[["shift_x", "shift_y", "shift_z"]])

    @property
    def shift_delta(self) -> np.ndarray:
        """
        Get the shift deltas (drift)

        """
        return np.array(self.data[["shift_delta_x", "shift_delta_y", "shift_delta_z"]])

    @property
    def beam_tilt_theta(self) -> np.ndarray:
        """
        Get the beam tilt theta angles

        """
        return self.data["beam_tilt_theta"]

    @property
    def beam_tilt_phi(self) -> np.ndarray:
        """
        Get the beam tilt phi angles

        """
        return self.data["beam_tilt_phi"]

    @property
    def electrons_per_angstrom(self) -> np.ndarray:
        """
        Get the dose

        """
        return self.data["electrons_per_angstrom"]

    @property
    def exposure_time(self) -> np.ndarray:
        """
        Get the exposure times

        """
        return self.data["exposure_time"]

    @property
    def position(self) -> np.ndarray:
        """
        Get the positions

        """
        return self.shift + self.shift_delta

    @property
    def angles(self) -> np.ndarray:
        """
        Get the angles

        """
        return np.degrees(self.data["angle"])

    @property
    def axes(self) -> np.ndarray:
        """
        Get the axes

        """
        return np.array(self.data[["axis_x", "axis_y", "axis_z"]])

    @property
    def defocus_offset(self) -> np.ndarray:
        """
        Get the defocus offset

        """
        return self.data["defocus_offset"]

    @property
    def euler_angles(self) -> np.ndarray:
        """
        Euler angle representation of the orientations.

        The Euler angles are intrinsic, right handed rotations around ZYZ.
        This matches the convention used by XMIPP/RELION.

        """
        return R.from_rotvec(self.orientation).inv().as_euler(seq="ZYZ", degrees=True)

    def __len__(self) -> int:
        """
        Get the number of scan points

        """
        return len(self.data)


class ScanFactory(object):
    """
    A Factory class to generate scans

    """

    @classmethod
    def _generate_drift(
        Class,
        angles: np.ndarray,
        x: Union[float, tuple] = 0,
        y: Union[float, tuple] = 0,
        z: Union[float, tuple] = 0,
    ) -> np.ndarray:
        """
        Get the beam drift

        """

        # Check type
        if isinstance(x, float):
            x = tuple([x, 0.0])
        if isinstance(y, float):
            y = tuple([y, 0.0])
        if isinstance(z, float):
            z = tuple([z, 0.0])

        # Compute the sigma of the normal in x, y, z
        x_sigma = x[0] + x[1] * np.radians(angles) ** 4
        y_sigma = y[0] + x[1] * np.radians(angles) ** 4
        z_sigma = z[0] + x[1] * np.radians(angles) ** 4

        # Generate some random noise
        drift = np.random.normal(0, [x_sigma, y_sigma, z_sigma]).T

        # Return the drift
        return drift

    @classmethod
    def _rotvec_from_axis_and_angles(
        Class, axis: np.ndarray, angles: np.ndarray
    ) -> tuple:
        """
        Generate the rotvec

        """
        axis = axis / np.linalg.norm(axis)
        return np.array([axis for a in angles]), angles

    @classmethod
    def _axis_angle_from_rotvec(Class, orientation) -> tuple:
        """
        Get the axis and angle from the rotvec

        """
        n = np.linalg.norm(orientation, axis=1)
        d = np.dot(orientation, np.array([0, 1, 0]))
        angle = np.degrees(n * np.sign(d))
        s = n > 0
        n[s] = 1.0 / n[s]
        n = n * np.sign(d)
        axis = orientation * n[:, np.newaxis]
        return axis, angle

    @classmethod
    def _shift_from_axis_and_positions(
        Class, axis: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """
        Generate the shifts

        """
        if len(positions.shape) == 1:
            positions = axis * positions[:, np.newaxis]
        else:
            assert len(positions.shape) == 2
            assert positions.shape[1] == 2
            z = np.zeros((positions.shape[0], 1))
            positions = np.append(positions, z, axis=1)
        return positions

    @classmethod
    def single_axis(
        Class,
        axis: Union[np.ndarray, tuple] = (0, 1, 0),
        angles: np.ndarray = None,
        positions: np.ndarray = None,
        defocus_offset: np.ndarray = None,
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Create a single axis scan

        """

        # Check input
        if angles is None:
            angles = np.array([])
        if positions is None:
            positions = np.array([])
        if defocus_offset is None:
            defocus_offset = np.zeros(len(angles))
        assert angles is not None
        assert positions is not None
        assert defocus_offset is not None
        assert len(angles) == len(positions)
        assert len(angles) == len(defocus_offset)
        num_images = len(angles)

        # Create the orientation and shift
        axis, angle = Class._rotvec_from_axis_and_angles(np.array(axis), angles)
        shift = Class._shift_from_axis_and_positions(np.array(axis), positions)

        # Set the image number and frame number
        image_number = np.arange(num_images)
        fraction_number = np.arange(num_fractions)
        fraction_number = np.repeat([fraction_number], num_images, axis=0).flatten()

        # Duplicate for the number of movie frames per step
        image_number = np.repeat(image_number, num_fractions, axis=0)
        axis = np.repeat(axis, num_fractions, axis=0)
        angle = np.repeat(angle, num_fractions, axis=0)
        shift = np.repeat(shift, num_fractions, axis=0)
        dose = np.full(angle.shape, electrons_per_angstrom / num_fractions)
        defocus_offset = np.repeat(defocus_offset, num_fractions, axis=0)

        # Create the shift delta
        shift_delta = None
        if drift is not None:
            shift_delta = Class._generate_drift(angles, **drift)
            shift_delta = np.repeat(shift_delta, num_fractions, axis=0)

        # Create the scan object
        return Scan(
            image_number=image_number,
            fraction_number=fraction_number,
            axis=np.array(axis),
            angle=angle,
            shift=shift,
            shift_delta=shift_delta,
            electrons_per_angstrom=dose,
            exposure_time=exposure_time,
            defocus_offset=defocus_offset,
        )

    @classmethod
    def manual(
        Class,
        axis: tuple = (0, 1, 0),
        angles: np.ndarray = None,
        positions: np.ndarray = None,
        defocus_offset: np.ndarray = None,
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Create a manual scan with custom angles and positions

        """
        # Check angles and positions
        if angles is None and positions is None:
            angles = np.array([0])
            positions = np.array([0])
        elif angles is None and positions is not None:
            angles = np.zeros(len(positions))
        elif positions is None and angles is not None:
            positions = np.zeros(len(angles))
        if defocus_offset is None:
            defocus_offset = np.zeros(len(angles))  # type: ignore
        assert angles is not None
        assert positions is not None
        assert len(angles) == len(positions)
        assert len(angles) == len(defocus_offset)
        angles = np.array(angles)
        positions = np.array(positions)

        # Create the single axis scan
        return Class.single_axis(
            axis=axis,
            angles=angles,
            positions=positions,
            defocus_offset=defocus_offset,
            num_fractions=num_fractions,
            electrons_per_angstrom=electrons_per_angstrom,
            exposure_time=exposure_time,
            drift=drift,
        )

    @classmethod
    def still(
        Class,
        axis: tuple = (0, 1, 0),
        start_angle: float = 0,
        start_pos: float = 0,
        num_images: int = 1,
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Create a still image scan

        """
        angles = np.repeat(start_angle, num_images)
        positions = np.repeat(start_pos, num_images)
        return Class.single_axis(
            axis=axis,
            angles=angles,
            positions=positions,
            num_fractions=num_fractions,
            electrons_per_angstrom=electrons_per_angstrom,
            exposure_time=exposure_time,
            drift=drift,
        )

    @classmethod
    def tilt_series(
        Class,
        axis: tuple = (0, 1, 0),
        start_angle: float = 0,
        step_angle: float = 0,
        start_pos: float = 0,
        num_images: int = 1,
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Create a continuous single axis scan

        """

        # Create list of angles and positions
        angles = start_angle + step_angle * np.arange(num_images)
        positions = np.full(len(angles), start_pos)

        # Create the scan
        return Class.single_axis(
            axis=axis,
            angles=angles,
            positions=positions,
            num_fractions=num_fractions,
            electrons_per_angstrom=electrons_per_angstrom,
            exposure_time=exposure_time,
            drift=drift,
        )

    @classmethod
    def dose_symmetric(
        Class,
        axis: tuple = (0, 1, 0),
        start_angle: float = 0,
        step_angle: float = 0,
        start_pos: float = 0,
        num_images: int = 1,
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Create a dose symmetric single axis scan

        """
        # Create the list of angles
        angles = start_angle + step_angle * np.arange(num_images)
        angles = np.array(sorted(angles, key=lambda x: abs(x)))
        positions = np.full(len(angles), start_pos)

        # Create the scan
        return Class.single_axis(
            axis=axis,
            angles=angles,
            positions=positions,
            num_fractions=num_fractions,
            electrons_per_angstrom=electrons_per_angstrom,
            exposure_time=exposure_time,
            drift=drift,
        )

    @classmethod
    def helical(
        Class,
        axis: tuple = (0, 1, 0),
        start_angle: float = 0,
        step_angle: float = 0,
        start_pos: float = 0,
        step_pos: float = 0,
        num_images: int = 1,
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Create a continous helical scan

        """
        # Create the list of angles and positions
        angles = start_angle + step_angle * np.arange(num_images)
        positions = start_pos + step_pos * np.arange(num_images)

        # Create a single axis scan
        return Class.single_axis(
            axis=axis,
            angles=angles,
            positions=positions,
            num_fractions=num_fractions,
            electrons_per_angstrom=electrons_per_angstrom,
            exposure_time=exposure_time,
            drift=drift,
        )

    @classmethod
    def nhelix(
        Class,
        axis: tuple = (0, 1, 0),
        start_angle: float = 0,
        step_angle: float = 0,
        start_pos: float = 0,
        step_pos: float = 0,
        num_images: int = 1,
        num_fractions: int = 1,
        num_nhelix: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Create a n-helix scan

        """
        # Create the list of angles and positions
        angles = np.zeros((num_nhelix, num_images))
        positions = np.zeros((num_nhelix, num_images))
        for j in range(num_nhelix):
            start_angle_j = start_angle + step_angle * j / num_nhelix
            angles[j, :] = start_angle_j + step_angle * np.arange(num_images)
            positions[j, :] = start_pos + np.full(num_images, j * step_pos)

        # Create a single axis scan
        return Class.single_axis(
            axis=axis,
            angles=angles.flatten(),
            positions=positions.flatten(),
            num_fractions=num_fractions,
            electrons_per_angstrom=electrons_per_angstrom,
            exposure_time=exposure_time,
            drift=drift,
        )

    @classmethod
    def single_particle(
        Class,
        num_images: int = 1,
        num_fractions: int = 1,
        exposure_time: float = 1,
        electrons_per_angstrom: float = 1,
        drift: dict = None,
        defocus_offset: np.ndarray = None,
        **kwargs
    ) -> Scan:
        """
        Create a single particle scan. This is a special scan that is actually
        just the particle in different orientations

        """
        if defocus_offset is None:
            defocus_offset = np.zeros(num_images)
        assert num_images == len(defocus_offset)

        # Get a random list of uniform orientations
        orientation = R.from_matrix(
            special_ortho_group.rvs(dim=3, size=num_images)
        ).as_rotvec()

        # Get the axis and angle
        axis, angle = Class._axis_angle_from_rotvec(orientation)

        # Set the image number and fraction number
        image_number = np.arange(len(angle))
        fraction_number = np.arange(num_fractions)
        fraction_number = np.repeat([fraction_number], len(angle), axis=0).flatten()

        # Duplicate for the number of movie frames per step
        image_number = np.repeat(image_number, num_fractions, axis=0)
        axis = np.repeat(axis, num_fractions, axis=0)
        angle = np.repeat(angle, num_fractions, axis=0)
        dose = np.full(angle.shape, electrons_per_angstrom / num_fractions)
        defocus_offset = np.repeat(defocus_offset, num_fractions, axis=0)

        # Create the shift delta
        shift_delta = None
        if drift is not None:
            shift_delta = Class._generate_drift(np.zeros(num_images), **drift)
            shift_delta = np.repeat(shift_delta, num_fractions, axis=0)

        # Create the scan
        return Scan(
            image_number=image_number,
            fraction_number=fraction_number,
            axis=axis,
            angle=angle,
            shift_delta=shift_delta,
            electrons_per_angstrom=dose,
            exposure_time=exposure_time,
            defocus_offset=defocus_offset,
            is_uniform_angular_scan=True,
        )

    @classmethod
    def beam_tilt(
        Class,
        axis: Union[np.ndarray, tuple] = (0, 1, 0),
        angles: np.ndarray = None,
        positions: np.ndarray = None,
        theta: np.ndarray = None,
        phi: np.ndarray = None,
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Make a beam tilt scan. For each (angle, position) we scan the beam tilt

        """

        # Check angles and positions
        if angles is None and positions is None:
            angles = np.array([0])
            positions = np.array([0])
        elif angles is None and positions is not None:
            angles = np.zeros(len(positions))
        elif positions is None and angles is not None:
            positions = np.zeros(len(angles))
        assert positions is not None
        assert angles is not None
        assert len(angles) == len(positions)

        # Check the beam tilt parameters
        if theta is None and phi is None:
            theta = np.array([0])
            phi = np.array([0])
        elif theta is None and phi is not None:
            theta = np.zeros(len(phi))
        elif phi is None and theta is not None:
            phi = np.zeros(len(theta))
        assert phi is not None
        assert theta is not None
        assert len(theta) == len(phi)

        # Make the beam_tilt * stage_tilt scan
        num_stage_tilt = len(angles)
        num_beam_tilt = len(theta)
        angles = np.stack([angles] * num_beam_tilt).T.flatten()
        positions = np.stack([positions] * num_beam_tilt).T.flatten()
        beam_tilt_theta = np.stack([theta] * num_stage_tilt).flatten()
        beam_tilt_phi = np.stack([phi] * num_stage_tilt).flatten()
        assert angles is not None
        assert positions is not None

        # Create the orientation and shift
        axis, angle = Class._rotvec_from_axis_and_angles(np.array(axis), angles)
        shift = Class._shift_from_axis_and_positions(np.array(axis), positions)

        # Set the image number and fraction number
        image_number = np.arange(len(angles))
        fraction_number = np.arange(num_fractions)
        fraction_number = np.repeat([fraction_number], len(angles), axis=0).flatten()

        # Duplicate for the number of movie frames per step
        image_number = np.repeat(image_number, num_fractions, axis=0)
        axis = np.repeat(axis, num_fractions, axis=0)
        angle = np.repeat(angle, num_fractions, axis=0)
        shift = np.repeat(shift, num_fractions, axis=0)
        beam_tilt_theta = np.repeat(beam_tilt_theta, num_fractions, axis=0)
        beam_tilt_phi = np.repeat(beam_tilt_phi, num_fractions, axis=0)
        dose = np.full(angle.shape, electrons_per_angstrom / num_fractions)

        # Create the shift delta
        shift_delta = None
        if drift is not None:
            shift_delta = Class._generate_drift(angles, **drift)
            shift_delta = np.repeat(shift_delta, num_fractions, axis=0)

        # Create the scan object
        return Scan(
            image_number=image_number,
            fraction_number=fraction_number,
            axis=np.array(axis),
            angle=angle,
            shift=shift,
            shift_delta=shift_delta,
            beam_tilt_theta=beam_tilt_theta,
            beam_tilt_phi=beam_tilt_phi,
            electrons_per_angstrom=dose,
            exposure_time=exposure_time,
        )

    @classmethod
    def grid_scan(
        Class,
        axis: Union[np.ndarray, tuple] = (0, 1, 0),
        angles: np.ndarray = None,
        start_pos: tuple = (0, 0),
        step_pos: tuple = (0, 0),
        num_images: tuple = (1, 1),
        num_fractions: int = 1,
        electrons_per_angstrom: float = 1,
        exposure_time: float = 1,
        drift: dict = None,
        **kwargs
    ) -> Scan:
        """
        Make a gridscan. For each angle we perform the grid scan

        """

        # Create the list of angles
        if angles is None:
            angles = np.array([0])

        # Create the list of positions
        x = start_pos[0] + step_pos[0] * np.arange(num_images[0])
        y = start_pos[1] + step_pos[1] * np.arange(num_images[1])
        positions = np.array([(xx, yy) for yy in x for xx in y])

        # Get the positions
        len_angles = len(angles)
        len_positions = len(positions)
        angles = np.repeat(angles, len_positions, axis=0)
        positions = np.concatenate([positions] * len_angles)

        # Create a single axis scan
        return Class.single_axis(
            axis=axis,
            angles=angles,
            positions=positions,
            num_fractions=num_fractions,
            electrons_per_angstrom=electrons_per_angstrom,
            exposure_time=exposure_time,
            drift=drift,
        )

    @classmethod
    def make_scan(Class, mode: str, **kwargs) -> Scan:
        """
        Make a scan from the input arguments

        """

        # Select the factory function
        function = {
            None: Class.manual,
            "manual": Class.manual,
            "still": Class.still,
            "tilt_series": Class.tilt_series,
            "dose_symmetric": Class.dose_symmetric,
            "helical_scan": Class.helical,
            "nhelix": Class.nhelix,
            "single_particle": Class.single_particle,
            "beam_tilt": Class.beam_tilt,
            "grid_scan": Class.grid_scan,
        }[mode]

        # Create the scan
        return function(**kwargs)  # type: ignore


def new(
    mode: str = "still",
    axis: tuple = (0, 1, 0),
    angles: np.ndarray = None,
    positions: np.ndarray = None,
    start_angle: float = 0,
    step_angle: float = 0,
    start_pos: float = 0,
    step_pos: float = 0,
    num_images: int = 1,
    num_fractions: int = 1,
    num_nhelix: int = 1,
    exposure_time: float = 1,
    defocus_offset: np.ndarray = None,
    theta: np.ndarray = None,
    phi: np.ndarray = None,
    drift: dict = None,
    electrons_per_angstrom: float = 40,
) -> Scan:
    """
    Create an scan

    If angles or positions is None they are generated form the other
    parameters.

    Args:
        mode: The type of scan (still, tilt_series, dose_symmetric, helical_scan)
        axis: The rotation axis
        angles: The rotation angles
        positions: The positions
        start_angle: The starting angle (deg)
        step_angle: The angle step (deg)
        start_pos: The starting position (A)
        step_pos: The step in position (A)
        num_images: The number of images
        num_fractions: The number of movie frames per image
        num_nhelix: The number of scans in an n-helix
        exposure_time: The exposure time (seconds)
        defocus_offset: The defocus_offset (A)
        theta: The beam tilt theta angle
        phi: The beam tilt phi angle
        drift: The beam drift model
        electrons_per_angstrom: The number of electrons per angstrom (per image)

    Returns:
        The scan object

    """
    kwargs = {
        "axis": axis,
        "angles": angles,
        "positions": positions,
        "start_angle": start_angle,
        "step_angle": step_angle,
        "start_pos": start_pos,
        "step_pos": step_pos,
        "num_images": num_images,
        "num_fractions": num_fractions,
        "num_nhelix": num_nhelix,
        "exposure_time": exposure_time,
        "defocus_offset": defocus_offset,
        "theta": theta,
        "phi": phi,
        "drift": drift,
        "electrons_per_angstrom": electrons_per_angstrom,
    }
    return ScanFactory.make_scan(mode, **kwargs)
