import numpy as np


def compute_phase_shift_for_freq(k, phase_shift=np.pi / 2, radius=0.005):
    """
    Compute the phase shift from a phase plate

    """
    # Multiply the wave with the phase shift from the phase plate which is
    # approximated by a phase shift applied only on the near field terms
    return np.exp(1j * phase_shift * (1 - np.exp(-(k**2) / (2 * radius**2))))


def compute_phase_shift(shape, pixel_size, phase_shift=np.pi / 2, radius=0.005):
    """
    Compute the phase shift from a phase plate

    """

    # Compute the spatial frequencies
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    Y = (Y - shape[0] // 2) / (pixel_size * shape[0])
    X = (X - shape[1] // 2) / (pixel_size * shape[1])
    k = np.sqrt(X**2 + Y**2)

    # Multiply the wave with the phase shift from the phase plate which is
    # approximated by a phase shift applied only on the near field terms
    return np.fft.ifftshift(compute_phase_shift_for_freq(k, phase_shift, radius))
