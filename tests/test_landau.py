import numpy as np
import pytest
from parakeet import landau


def test_electron_velocity():
    v = landau.electron_velocity(300000)
    assert v == pytest.approx(0.7765254931159066)


def test_landau():
    psi = landau.landau(0)
    assert psi == pytest.approx(0.178858382958185)
    psi = landau.landau(-0.223)
    assert psi == pytest.approx(0.18065985645723706)


def test_mpl_and_fwhm():
    mpl, fwhm = landau.mpl_and_fwhm(300, 3150)
    assert mpl == pytest.approx(17.928069663617993)
    assert fwhm == pytest.approx(12.480772265960773)


def test_energy_loss_distribution():
    psi = landau.energy_loss_distribution(np.arange(0, 100), 300, 3150)
    assert np.argmax(psi) == 18
