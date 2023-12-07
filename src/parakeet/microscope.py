#
# parakeet.microscope.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy as np
import parakeet.config
import parakeet.beam
import parakeet.detector
import parakeet.lens
from typing import Optional
from math import pi


class PhasePlate(object):
    """
    A class to encapsulate a phase plate

    """

    def __init__(self, use=False, phase_shift=0.5 * pi, radius=0.005):
        """
        Init the phase plate

        """
        self.use = use
        self.phase_shift = phase_shift
        self.radius = radius


class Microscope(object):
    """
    A class to encapsulate a microscope

    """

    def __init__(
        self,
        model: parakeet.config.MicroscopeModel = None,
        beam: parakeet.beam.Beam = parakeet.beam.Beam(),
        lens: parakeet.lens.Lens = parakeet.lens.Lens(),
        detector: parakeet.detector.Detector = parakeet.detector.Detector(),
        phase_plate: PhasePlate = PhasePlate(),
        objective_aperture_cutoff_freq=None,
    ):
        """
        Initialise the detector

        Args:
            model: The microscope model name
            beam: The beam object
            lens: The lens object
            detector: The detector object
            phase_plate: The phase plate

        """
        self._model = model
        self._beam = beam
        self._lens = lens
        self._detector = detector
        self._phase_plate = phase_plate
        self.objective_aperture_cutoff_freq = objective_aperture_cutoff_freq

    @property
    def model(self) -> Optional[parakeet.config.MicroscopeModel]:
        """
        The microscope model type

        """
        return self._model

    @property
    def beam(self) -> parakeet.beam.Beam:
        """
        The beam model

        """
        return self._beam

    @property
    def lens(self) -> parakeet.lens.Lens:
        """
        The lens model

        """
        return self._lens

    @property
    def detector(self) -> parakeet.detector.Detector:
        """
        The detector model

        """
        return self._detector

    @property
    def phase_plate(self) -> PhasePlate:
        """
        Do we have a phase plate

        """
        return self._phase_plate


def new(
    config: parakeet.config.Microscope,
) -> Microscope:
    """
    Make a new microscope object

    Args:
        config: The microscope model configuration

    Returns:
        The microscope model

    """

    # Construct the basic models from the input
    beam = parakeet.beam.new(config.beam)
    lens = parakeet.lens.new(config.lens)
    detector = parakeet.detector.new(config.detector)

    # Override the parameters for the different microscope models
    if config.model == "krios":
        beam.energy = 300
        beam.energy_spread = 2.66 * 1e-6
        beam.acceleration_voltage_spread = 0.8 * 1e-6
        lens.c_30 = 2.7
        lens.c_c = 2.7
        lens.current_spread = 0.33 * 1e-6
    elif config.model == "talos":
        beam.energy = 200
        beam.energy_spread = 2.66 * 1e-6
        beam.acceleration_voltage_spread = 0.8 * 1e-6
        lens.c_30 = 2.7
        lens.c_c = 2.7
        lens.current_spread = 0.33 * 1e-6
    elif config.model is not None:
        raise RuntimeError("Unknown microscope model")

    # Return the miroscope object
    return Microscope(
        model=config.model,
        beam=beam,
        lens=lens,
        detector=detector,
        phase_plate=PhasePlate(
            config.phase_plate.use,
            np.radians(config.phase_plate.phase_shift),
            config.phase_plate.radius,
        ),
        objective_aperture_cutoff_freq=config.objective_aperture_cutoff_freq,
    )
