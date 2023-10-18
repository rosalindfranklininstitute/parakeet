#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import parakeet.sample
import argparse
import multem
import mrcfile
import numpy as np
import os.path
import pandas
import scipy.constants
import urllib.request
from math import ceil, pi, log
from matplotlib import pylab


def get_water_model_filename():
    """
    Get the water model filename

    """
    return "water_645_coords.pdb"


def get_water_atomic_model():
    """
    Download the water model from zenodo

    """
    filename = get_water_model_filename()
    if not os.path.exists(filename):
        print("Downloading %s" % filename)
        link = "https://zenodo.org/record/4415836/files/%s?download=1" % filename
        urllib.request.urlretrieve(link, filename)


def load_water_atomic_model():
    """
    Extract the water atomic model to a file

    """
    # Read the atom data
    if not os.path.exists("atoms.csv"):
        print("Reading model from %s" % get_water_model_filename())
        atom_data = parakeet.sample.AtomData.from_gemmi_file(get_water_model_filename())
        atom_data.data.to_csv("atoms.csv")
    else:
        print("Reading model from %s" % "atoms.csv")
        atom_data = parakeet.sample.AtomData(data=pandas.read_csv("atoms.csv"))
    return atom_data


def next_power_2(x):
    """
    Get the next power of two

    """
    return 2 ** (ceil(log(x, 2)))


def radial_average(data):
    """
    Compute the radial average

    """
    Y, X = np.indices((data.shape))
    ysize, xsize = data.shape
    distance = np.sqrt((X - xsize / 2.0) ** 2 + (Y - ysize / 2.0) ** 2)
    distance = np.floor(distance).astype("int32")
    c1 = np.bincount(distance.ravel()).astype("float32")
    c2 = np.bincount(distance.ravel(), data.ravel()).astype("float32")
    return c2 / c1


def create_input_multislice():
    """
    Create the input multislice parameters

    """

    # Initialise the input and system configuration
    input_multislice = multem.Input()

    # Set simulation experiment
    input_multislice.simulation_type = "EWRS"

    # Electron-Specimen interaction model
    input_multislice.interaction_model = "Multislice"
    # input_multislice.potential_type = "Lobato_0_12"
    input_multislice.potential_type = "Peng_0_12"

    # Potential slicing
    input_multislice.potential_slicing = "dz_Proj"

    # Electron-Phonon interaction model
    input_multislice.pn_model = "Still_Atom"

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    # Set the energy
    input_multislice.E_0 = 300
    input_multislice.bwl = False

    return input_multislice


def compute_potential():
    """
    Compute the projected potential in slices

    """

    def compute(atom_data, pixel_size, thickness):
        # Get the dimensions
        x_min = atom_data.data["x"].min()
        x_max = atom_data.data["x"].max()
        y_min = atom_data.data["y"].min()
        y_max = atom_data.data["y"].max()
        z_min = atom_data.data["z"].min()
        z_max = atom_data.data["z"].max()
        x_size = x_max - x_min
        y_size = y_max - y_min
        z_size = z_max - z_min

        # Translate to centre
        x_box_size = x_size
        y_box_size = y_size
        z_box_size = z_size
        x_trans = (x_box_size - x_size) / 2.0 - x_min
        y_trans = (y_box_size - y_size) / 2.0 - y_min
        z_trans = (z_box_size - z_size) / 2.0 - z_min
        atom_data.translate((x_trans, y_trans, z_trans))

        # Trim the atom data
        z_select_min = z_box_size / 2 - thickness / 2
        z_select_max = z_box_size / 2 + thickness / 2
        selection = (atom_data.data["z"] > z_select_min) & (
            atom_data.data["z"] < z_select_max
        )
        atom_data = parakeet.sample.AtomData(data=atom_data.data[selection])
        num_atoms = len(atom_data.data)

        # Create the system configuration
        system_conf = multem.SystemConfiguration()
        system_conf.precision = "float"
        system_conf.device = "device"

        # Create the input multislice configuration
        input_multislice = create_input_multislice()

        # Compute the number of pixels
        nx = int(x_box_size / pixel_size)
        ny = int(y_box_size / pixel_size)
        x_box_size = nx * pixel_size
        y_box_size = ny * pixel_size

        # Create the specimen size
        input_multislice.nx = nx
        input_multislice.ny = ny
        input_multislice.spec_lx = x_box_size
        input_multislice.spec_ly = y_box_size
        input_multislice.spec_lz = z_box_size
        input_multislice.spec_dz = thickness

        # Set the specimen atoms
        input_multislice.spec_atoms = atom_data.to_multem()

        potential = []

        def callback(z0, z1, V):
            V = np.array(V)
            potential.append(V)

        # Run the simulation
        multem.compute_projected_potential(system_conf, input_multislice, callback)

        # Save the potential
        potential = np.sum(potential, axis=0)
        filename = "potential_%.1f_%d.npz" % (pixel_size, thickness)
        np.savez(filename, potential=potential, num_atoms=num_atoms)

    # Read the atom data
    atom_data = load_water_atomic_model()

    # Simulate the projected potential
    for pixel_size in np.arange(0.1, 2.1, 0.1):
        for thickness in np.arange(5, 25, 5):
            print("Compute potential for px = %f, dz = %f" % (pixel_size, thickness))
            if not os.path.exists("potential_%.1f_%d.npz" % (pixel_size, thickness)):
                compute(atom_data, pixel_size, thickness)


def compute_observed_mean(size, pixel_size):
    """
    Compute the observed mean

    """

    # Create the system configuration
    system_conf = multem.SystemConfiguration()
    system_conf.precision = "float"
    system_conf.device = "device"

    # Create the input multislice configuration
    input_multislice = create_input_multislice()

    # Compute the number of pixels
    nx = int(ceil(size / pixel_size))
    ny = int(ceil(size / pixel_size))
    size = nx * pixel_size

    # Create the specimen atoms
    input_multislice.nx = nx
    input_multislice.ny = ny
    input_multislice.spec_lx = nx * pixel_size
    input_multislice.spec_ly = ny * pixel_size
    input_multislice.spec_lz = nx * pixel_size
    input_multislice.spec_dz = 1

    # For N random placements compute the mean intensity
    means = []
    for j in range(10):
        # Compute the position
        x0 = np.random.uniform(0, 1) + nx // 2
        y0 = np.random.uniform(0, 1) + ny // 2
        x0 = pixel_size * x0
        y0 = pixel_size * y0

        # Set the atom list
        input_multislice.spec_atoms = multem.AtomList(
            [
                (1, x0, y0, size / 2.0, 0, 1, 0, 0),
                (1, x0, y0, size / 2.0, 0, 1, 0, 0),
                (8, x0, y0, size / 2.0, 0, 1, 0, 0),
            ]
        )

        thickness = []
        potential = []

        def callback(z0, z1, V):
            V = np.array(V)
            thickness.append(z1 - z0)
            potential.append(V)

        # Run the simulation
        multem.compute_projected_potential(system_conf, input_multislice, callback)

        # Compute the mean potential
        V = np.sum(potential, axis=0)
        means.append(np.mean(V))

    # Return the size and mean potential
    return size, np.mean(means)


def compute_expected_mean(size):
    """
    Compute the expected mean

    """

    def compute(size, Z):
        hbar = scipy.constants.hbar
        qe = scipy.constants.e
        me = scipy.constants.electron_mass
        # P = multem.compute_V_params("Lobato_0_12", Z, 0)
        P = multem.compute_V_params("Peng_0_12", Z, 0)
        mean = 0
        for i in range(len(P)):
            Ai = P[i][0]
            Bi = P[i][1]
            if Bi > 0:
                k = me * qe / (2 * pi * hbar**2)  # C m^-2 s
                k = 1.0 / (k * 1e-10**2)  # C A^-2 s
                Bj = pi**2 / Bi
                Aj = Ai * k * (Bj / pi) ** (3.0 / 2.0)
                # Bj = (2*pi/Bi)**2
                # Aj = 2*Ai*k*(Bj)**(3.0/2.0)/pi**2
                mean += (1 / k) * Aj
        N = 1
        mean = mean * N / (size**2)
        return mean

    # Return the mean
    return compute(size, 1) * 2 + compute(size, 8)


def compute_mean_correction(ax=None):
    """
    Compute the mean correction table

    """

    size = 40

    pixel_size = np.arange(0.1, 2.1, 0.1)
    mean = []
    for ps in pixel_size:
        s, m = compute_observed_mean(size, ps)
        expected = compute_expected_mean(s)
        m = m / expected
        mean.append(m)

    area = [0] + list(pixel_size**2)
    mean = [1] + list(mean)

    # Write out the table to file
    print("Mean0 = %f" % (expected * s**2))
    with open("mean_table.csv", "w") as outfile:
        for x, y in zip(area, mean):
            outfile.write("%.2f, %.7f\n" % (x, y))

    # Plot the mean correction
    ax.plot(area, mean, label="Mean correction")
    ax.set_xlabel(r"Pixel area ($Å^2$)" + "\n(c)", fontsize=9)
    ax.set_title("Mean correction factor", fontsize=9)
    ax.set_xlim(0, 4)


def compute_mean_correction2(ax=None):
    """
    Compute the mean correction table

    """

    X = []
    Y = []

    # Loop through the pixel sizes
    thickness = 20
    for pixel_size in np.arange(0.1, 2.1, 0.1):
        # Read the projected potential
        handle = np.load("potential_%.1f_%d.npz" % (pixel_size, thickness))
        potential = handle["potential"]
        num_atoms = handle["num_atoms"]

        # Select the pixels with potential in
        xsize, ysize = potential.shape
        x0 = 0  # xsize // 8
        x1 = xsize  # 7 * x0
        y0 = 0  # ysize // 8
        y1 = ysize  # 7 * y0
        potential = potential[x0:x1, y0:y1]

        # Compute the density and variance
        num_molecules = num_atoms / 3.0
        ny, nx = potential.shape
        area = (nx * pixel_size) * (ny * pixel_size)
        mean = np.mean(potential)
        density = num_molecules / (area)

        # Append the pixel area and variance / density
        X.append(pixel_size**2)
        Y.append(mean / density)

    X = np.array(X)
    Y = np.array(Y)

    # Extrapolate to zero and normalize
    Y0 = Y[0] - X[0] * (Y[1] - Y[0]) / (X[1] - X[0])
    Y = Y / Y0

    X = [0] + list(X)
    Y = [1.0] + list(Y)

    # Write out the table to file
    print("Mean0 = %f" % Y0)
    with open("mean_table2.csv", "w") as outfile:
        for x, y in zip(X, Y):
            outfile.write("%.2f, %.7f\n" % (x, y))

    # Plot
    ax.plot(X, Y)
    ax.set_xlabel(r"Pixel area ($Å^2$)" + "\n(d)", fontsize=9)
    ax.set_title("Mean correction factor", fontsize=9)
    ax.set_xlim(0, 4)


def compute_variance_correction(ax=None):
    """
    Compute the variance correction table

    """

    X = []
    Y = []

    # Loop through the pixel sizes
    thickness = 20
    for pixel_size in np.arange(0.1, 2.1, 0.1):
        # Read the projected potential
        handle = np.load("potential_%.1f_%d.npz" % (pixel_size, thickness))
        potential = handle["potential"]
        num_atoms = handle["num_atoms"]

        # Select the pixels with potential in
        xsize, ysize = potential.shape
        x0 = 0  # xsize // 8
        x1 = xsize  # 7 * x0
        y0 = 0  # ysize // 8
        y1 = ysize  # 7 * y0
        potential = potential[x0:x1, y0:y1]

        # Compute the density and variance
        num_molecules = num_atoms / 3.0
        ny, nx = potential.shape
        area = (nx * pixel_size) * (ny * pixel_size)
        var = np.var(potential)
        density = num_molecules / (area)

        # Append the pixel area and variance / density
        X.append(pixel_size**2)
        Y.append(var / density)

    X = np.array(X)
    Y = np.array(Y)

    # Extrapolate to zero and normalize
    Y0 = Y[0] - X[0] * (Y[1] - Y[0]) / (X[1] - X[0])
    Y = Y / Y0

    X = [0] + list(X)
    Y = [1.0] + list(Y)

    # Write out the table to file
    print("Var0 = %f" % Y0)
    with open("variance_table.csv", "w") as outfile:
        for x, y in zip(X, Y):
            outfile.write("%.2f, %.7f\n" % (x, y))

    # Plot
    ax.plot(X, Y, label="Variance correction")
    ax.set_xlabel(r"Pixel area ($Å^2$)" + "\n(c)", fontsize=9)
    ax.set_title("Mean and variance correction factors", fontsize=9)
    ax.set_xlim(0, 4)


def compute_power(ax=None):
    """
    Compute the power spectrum

    """

    # Compute the fit to the power spectrum
    pixel_size = 0.1
    for thickness in [20]:  # , 19, 18, 15, 10, 5]:
        # Read the projected potential
        handle = np.load("potential_%.1f_%d.npz" % (pixel_size, thickness))
        potential = handle["potential"]
        num_atoms = handle["num_atoms"]

        # Select the pixels with potential in
        xsize, ysize = potential.shape
        x0 = xsize // 8
        x1 = 7 * xsize // 8
        y0 = ysize // 8
        y1 = 7 * ysize // 8
        potential = potential[x0:x1, y0:y1]

        # Compute the density and variance
        num_molecules = num_atoms / 3.0
        ny, nx = potential.shape
        # area = (nx * pixel_size) * (ny * pixel_size)
        var = np.var(potential)
        mean = np.mean(potential)
        # density = num_molecules / (area)
        potential -= mean
        print("Mean: %.3f; Variance: %.3f" % (mean, var))

        # Compute the FFT of the data and the power spectrum
        fft_data = np.fft.fft2(potential)
        power = np.abs(fft_data) ** 2
        Y, X = np.mgrid[0 : power.shape[0], 0 : power.shape[1]]
        q = (1 / pixel_size) * np.sqrt(
            ((X - power.shape[1] / 2) / power.shape[1]) ** 2
            + ((Y - power.shape[0] / 2) / power.shape[0]) ** 2
        )
        q = np.fft.fftshift(q)

        def func(q, A0, A1, A2, A3):
            M = 1.0 / 2.88
            P = A0 * np.exp(-0.5 * q**2 / A1**2) + A2 * np.exp(
                -0.5 * (q - M) ** 2 / A3**2
            )
            # I = (2*pi*(A0*A1**2 + A2*M*sqrt(2*pi*A3**2)))
            # P /= I
            model = P
            model[0, 0] = normalized_power[0, 0]
            return model

        # def func2(q, *p):
        #     q = q.reshape(power.shape)
        #     return func(q, *p).flatten()

        # def residuals(p, q, normalized_power):
        #     A0, A1, A2, A3 = p
        #     A0 = np.abs(A0)
        #     A2 = np.abs(A2)
        #     # A0 = 0.1608
        #     # A2, A3 = 0.822372155, 0.08153797

        #     # A1, A3 = 0.68518427, 0.08693241
        #     M = 1.0 / 2.88
        #     P = A0 * np.exp(-0.5 * q ** 2 / A1 ** 2) + A2 * np.exp(
        #         -0.5 * (q - M) ** 2 / A3 ** 2
        #     )
        #     # I = (2*pi*(A0*A1**2 + A2*M*sqrt(2*pi*A3**2)))
        #     # P /= I
        #     model = P
        #     model[0, 0] = normalized_power[0, 0]
        #     W = 1.0 / (q * power.shape[0] + 1)
        #     A = np.sum(W * (model - normalized_power) ** 2) / np.sum(W)
        #     B = (np.sum(model) - np.sum(normalized_power)) ** 2 / model.size
        #     print(p, A, B)
        #     return A + B

        # Compute the variance correction factor
        Cv = np.exp(-3.2056 * pixel_size**2)
        C = Cv * num_molecules / (pixel_size**4)

        # Compute the total integral of the power
        I = np.sum(power) * (1 / pixel_size) ** 2 / power.size
        normalized_power = power / I

        # Fit a model to the normalized power
        params = [0.19465002, 0.7312113, 0.78343527, 0.08078005]
        # results = scipy.optimize.minimize(residuals, x0=params, args=(q, normalized_power))#q.flatten(), normalized_power.flatten(), p0=params, sigma=(q*power.shape[0]+1).flatten())
        ##params, _ = scipy.optimize.curve_fit(func2, q.flatten(), normalized_power.flatten(), p0=params, sigma=(q*power.shape[0]+1).flatten())
        # params = results.x
        params[0] = params[0] * I / C
        params[2] = params[2] * I / C
        print("Parameters:", params)

        # Compute the model
        model = func(q, *params)
        model = model * C

        # Compute the radial spectrum
        rp = radial_average(np.fft.fftshift(power))
        rm = radial_average(np.fft.fftshift(model))
        d = np.arange(rp.size) / (pixel_size * power.shape[0])

        # Plot the power spectrum and best fit
        ax.plot(d[1:], rp[1:] / C)  # , label="%d" % thickness)
        ax.plot(d[1:], rm[1:] / C, color="black", alpha=0.5, label="Model")

    # Set some plot properties
    ax.set_xlabel("Spatial frequency ($Å^{-1}$)\n(a)", fontsize=9)
    ax.set_title("Power spectrum", fontsize=9)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.set_yticklabels("")


def compute_correlation(ax=None):
    """
    Compute the correlation length

    """

    pixel_size = 0.1
    for thickness in [5]:  # , 19, 18, 15, 10, 5]:
        # Read the projected potential
        handle = np.load("potential_%.1f_%d.npz" % (pixel_size, thickness))
        potential = handle["potential"]
        num_atoms = handle["num_atoms"]

        # Select the pixels with potential in
        xsize, ysize = potential.shape
        x0 = xsize // 8
        x1 = 7 * xsize // 8
        y0 = ysize // 8
        y1 = 7 * ysize // 8
        potential = potential[x0:x1, y0:y1]

        data = potential
        data = data - np.mean(data)
        f = np.fft.fft2(data)
        power = np.abs(f) ** 2
        corr = np.abs(np.fft.ifft2(power))
        corr = np.fft.fftshift(corr)
        corr = corr / (corr.size * np.var(data))

        rc = radial_average(corr)[0 : data.shape[0] // 2]
        rd = np.arange(rc.size) * pixel_size
        rc = rc / np.max(rc)

        corr_length = 0
        val = rc[0] / np.exp(1)
        for d, c in zip(rd, rc):
            if c < val:
                corr_length = d
                break

    print("Correlation Length: %f" % corr_length)

    # Plot the correlation
    if ax is None:
        width = 0.0393701 * 190
        height = width * 0.74
        fig, ax = pylab.subplots(figsize=(width, height), constrained_layout=True)
    ax.plot(rd, rc)
    ax.axvline(
        corr_length,
        color="black",
        linestyle="--",
        label="$𝜉 = %.1f Å$" % corr_length,
    )

    # Set some plot properties
    ax.set_xlabel("Distance ($Å$)\n(b)", fontsize=9)
    ax.set_title("Autocorrelation", fontsize=9)
    ax.set_xlim(0, 2.5)
    ax.legend(fontsize=9)
    if ax is None:
        fig.savefig("correlation_%.1fA.png" % pixel_size, dpi=300, bbox_inches="tight")
        pylab.close("all")


def calibrate():
    """
    Calibrate ice model

    """
    # Get the water atomic model file
    get_water_atomic_model()

    # Compute the projected potential
    compute_potential()

    # Setup the figure
    width = 0.0393701 * 190
    height = width / 3.0
    fig, ax = pylab.subplots(ncols=3, figsize=(width, height), constrained_layout=True)
    compute_power(ax[0])
    compute_correlation(ax[1])
    compute_mean_correction(ax[2])
    # compute_mean_correction2(ax[1])
    compute_variance_correction(ax[2])
    ax[2].legend(fontsize=9)
    for axx in ax:
        axx.tick_params(axis="both", labelsize=9)
    fig.savefig("model.png", dpi=300, bbox_inches="tight")


def compute_exit_wave(atom_data, pixel_size):
    """
    Compute the exit wave

    """

    # Get the dimensions
    x_min = atom_data.data["x"].min()
    x_max = atom_data.data["x"].max()
    y_min = atom_data.data["y"].min()
    y_max = atom_data.data["y"].max()
    x_size = x_max - x_min
    y_size = y_max - y_min
    select = (
        (atom_data.data["x"] > x_min + x_size / 6)
        & (atom_data.data["x"] < x_max - x_size / 6)
        & (atom_data.data["y"] > y_min + y_size / 6)
        & (atom_data.data["y"] < y_max - y_size / 6)
    )
    atom_data = parakeet.sample.AtomData(data=atom_data.data[select])
    x_min = atom_data.data["x"].min()
    x_max = atom_data.data["x"].max()
    y_min = atom_data.data["y"].min()
    y_max = atom_data.data["y"].max()
    z_min = atom_data.data["z"].min()
    z_max = atom_data.data["z"].max()
    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    # Translate to centre
    x_box_size = x_size
    y_box_size = y_size
    z_box_size = z_size

    # Create the system configuration
    system_conf = multem.SystemConfiguration()
    system_conf.precision = "float"
    system_conf.device = "device"

    # Create the input multislice configuration
    input_multislice = create_input_multislice()

    # Compute the number of pixels
    nx = int(ceil(x_box_size / pixel_size) * 2)
    ny = int(ceil(y_box_size / pixel_size) * 2)

    x_box_size = nx * pixel_size
    y_box_size = ny * pixel_size
    x_trans = (x_box_size - x_size) / 2.0 - x_min
    y_trans = (y_box_size - y_size) / 2.0 - y_min
    z_trans = (z_box_size - z_size) / 2.0 - z_min
    atom_data.translate((x_trans, y_trans, z_trans))

    # Create the specimen size
    input_multislice.nx = nx
    input_multislice.ny = ny
    input_multislice.spec_lx = x_box_size
    input_multislice.spec_ly = y_box_size
    input_multislice.spec_lz = z_box_size
    input_multislice.spec_dz = 10

    # Set the specimen atoms
    input_multislice.spec_atoms = atom_data.to_multem()

    # Run the simulation
    output_multislice = multem.simulate(system_conf, input_multislice)

    # Get the image
    physical_image = np.array(output_multislice.data[0].psi_coh).T

    # Create the masker
    masker = multem.Masker(input_multislice.nx, input_multislice.ny, pixel_size)

    # Create the size of the cuboid
    masker.set_cuboid(
        (
            x_box_size / 2 - x_size / 2,
            y_box_size / 2 - y_size / 2,
            z_box_size / 2 - z_size / 2,
        ),
        (x_size, y_size, z_size),
    )

    # Run the simulation
    input_multislice.spec_atoms = multem.AtomList()
    output_multislice = multem.simulate(system_conf, input_multislice, masker)

    # Get the image
    random_image = np.array(output_multislice.data[0].psi_coh).T

    # Return the images
    x0 = np.array((x_box_size / 2 - x_size / 2, y_box_size / 2 - y_size / 2))
    x1 = np.array((x_box_size / 2 + x_size / 2, y_box_size / 2 + y_size / 2))
    return physical_image, random_image, x0, x1


def load_exit_wave(atom_data, ps):
    physical_filename = "exit_wave_physical_%.1f.mrc" % ps
    random_filename = "exit_wave_random_%.1f.mrc" % ps
    metadata_filename = "metadata_%.1f.dat" % ps

    if (
        not os.path.exists(physical_filename)
        or not os.path.exists(random_filename)
        or not os.path.exists(metadata_filename)
    ):
        print("Simulating for pixel size: %.1f A" % ps)
        physical_image, random_image, x0, x1 = compute_exit_wave(atom_data, ps)
        physical_image_file = mrcfile.new(physical_filename, overwrite=True)
        random_image_file = mrcfile.new(random_filename, overwrite=True)
        physical_image_file.set_data(physical_image.astype("complex64"))
        random_image_file.set_data(random_image.astype("complex64"))
        with open(metadata_filename, "w") as outfile:
            outfile.write("%f,%f,%f,%f" % (x0[0], x0[1], x1[0], x1[1]))
    else:
        print("Reading for pixel size: %.1f A" % ps)
        physical_image = mrcfile.open(physical_filename).data
        random_image = mrcfile.open(random_filename).data
        with open(metadata_filename) as infile:
            x00, x01, x10, x11 = map(float, infile.read().split(","))
            x0 = (x00, x01)
            x1 = (x10, x11)

    return physical_image, random_image, x0, x1


def plot_mean_and_var(physical_data, random_data, xmin, xmax, ps):
    x0 = np.floor(xmin / ps).astype("int32")
    x1 = np.floor(xmax / ps).astype("int32")
    xr = x1 - x0
    x0 = x0 + xr // 3
    x1 = x1 - xr // 3

    random_middle = random_data[x0[0] : x1[0], x0[1] : x1[1]]
    physical_middle = physical_data[x0[0] : x1[0], x0[1] : x1[1]]

    physical_middle_real = physical_middle.flatten().real
    physical_middle_imag = physical_middle.flatten().imag
    random_middle_real = random_middle.flatten().real
    random_middle_imag = random_middle.flatten().imag

    # physical_middle_real = np.angle(physical_middle.flatten())
    # physical_middle_imag = np.angle(physical_middle.flatten())
    # random_middle_real = np.abs(random_middle.flatten())
    # random_middle_imag = np.abs(random_middle.flatten())

    physical_middle_mean_real = np.mean(physical_middle_real)
    physical_middle_mean_imag = np.mean(physical_middle_imag)
    random_middle_mean_real = np.mean(random_middle_real)
    random_middle_mean_imag = np.mean(random_middle_imag)

    physical_middle_std_real = np.std(physical_middle_real)
    physical_middle_std_imag = np.std(physical_middle_imag)
    random_middle_std_real = np.std(random_middle_real)
    random_middle_std_imag = np.std(random_middle_imag)

    width = 0.0393701 * 190
    height = width * 0.75
    fig, ax = pylab.subplots(
        figsize=(width, height),
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax[0][0].hist(physical_middle.flatten().real, bins=20, density=True)
    ax[0][1].hist(physical_middle.flatten().imag, bins=20, density=True)
    ax[1][0].hist(random_middle.flatten().real, bins=20, density=True)
    ax[1][1].hist(random_middle.flatten().imag, bins=20, density=True)
    ax[0][0].set_title("Real component", fontweight="bold")
    ax[0][1].set_title("Imaginary component", fontweight="bold")
    ax[0][0].set_ylabel("Physical model", fontweight="bold")
    ax[1][0].set_ylabel("Random model", fontweight="bold")
    ax[0][0].set_xlabel("(a)")
    ax[0][1].set_xlabel("(b)")
    ax[1][0].set_xlabel("(c)")
    ax[1][1].set_xlabel("(d)")
    ax[0][0].axvline(physical_middle_mean_real, color="black")
    ax[0][1].axvline(physical_middle_mean_imag, color="black")
    ax[1][0].axvline(random_middle_mean_real, color="black")
    ax[1][1].axvline(random_middle_mean_imag, color="black")
    # ymax = ax[0][0].get_ylim()[1]
    if ps == 1.0:
        xr = 0.8
        xi = 0.4
    else:
        xr = 0.5
        xi = -1.0
    ax[0][0].text(
        xr,
        0.5 * ax[0][0].get_ylim()[1],
        "mean: %.2f\n sdev: %.2f"
        % (physical_middle_mean_real, physical_middle_std_real),
    )
    ax[0][1].text(
        xi,
        0.5 * ax[0][1].get_ylim()[1],
        "mean: %.2f\n sdev: %.2f"
        % (physical_middle_mean_imag, physical_middle_std_imag),
    )
    ax[1][0].text(
        xr,
        0.5 * ax[1][0].get_ylim()[1],
        "mean: %.2f\n sdev: %.2f" % (random_middle_mean_real, random_middle_std_real),
    )
    ax[1][1].text(
        xi,
        0.5 * ax[1][1].get_ylim()[1],
        "mean: %.2f\n sdev: %.2f" % (random_middle_mean_imag, random_middle_std_imag),
    )
    fig.savefig("histograms_%.1fA.png" % ps, dpi=300, bbox_inches="tight")
    pylab.close("all")

    return (
        physical_middle_mean_real,
        physical_middle_mean_imag,
        physical_middle_std_real,
        physical_middle_std_imag,
        random_middle_mean_real,
        random_middle_mean_imag,
        random_middle_std_real,
        random_middle_std_imag,
    )


def plot_power(physical_data, random_data, xmin, xmax, ps):
    x0 = np.floor(xmin / ps).astype("int32")
    x1 = np.floor(xmax / ps).astype("int32")
    xr = x1 - x0
    x0 = x0 + xr // 3
    x1 = x1 - xr // 3

    random_middle = random_data[x0[0] : x1[0], x0[1] : x1[1]]
    physical_middle = physical_data[x0[0] : x1[0], x0[1] : x1[1]]
    physical_middle_mean_real = np.mean(physical_middle.flatten().real)
    physical_middle_mean_imag = np.mean(physical_middle.flatten().imag)
    random_middle_mean_real = np.mean(random_middle.flatten().real)
    random_middle_mean_imag = np.mean(random_middle.flatten().imag)

    # pylab.imshow(np.abs(random_data))
    # pylab.show()
    # pylab.imshow(np.abs(physical_data))
    # pylab.show()

    def compute_power(data, pixel_size):
        f = np.fft.fft2(data)
        p = np.abs(f) ** 2
        p = np.fft.fftshift(p)

        r = radial_average(p)[0 : data.shape[0] // 2]
        d = np.arange(r.size) / (pixel_size * data.shape[0])

        N = np.mean(r[d < 0.5][1:])

        return d[1:], r[1:] / N

    random_d, random_power = compute_power(random_middle, ps)
    physical_d, physical_power = compute_power(physical_middle, ps)

    width = 0.0393701 * 190
    height = width * 0.74
    fig, ax = pylab.subplots(figsize=(width, height), constrained_layout=True)
    ax.plot(physical_d, physical_power, label="Physical model")
    ax.plot(random_d, random_power, label="Random model")
    ax.set_xlabel("Spatial frequency (1/Å)")
    ax.set_ylabel("Power spectrum")
    # ax.set_xlim(0, 1.0)
    ax.legend()
    fig.savefig("power_%.1fA.png" % ps, dpi=300, bbox_inches="tight")
    pylab.close("all")

    return physical_d, physical_power, random_d, random_power


def plot_edge(physical_data, random_data, xmin, xmax, ps):
    x0 = np.floor(xmin / ps).astype("int32")
    x1 = np.floor(xmax / ps).astype("int32")
    xd = x1 - x0
    x0 = x0 - xd // 10
    x1 = x0 + 3 * xd // 10

    # x1 = 2 * x0  # + x0 // 2
    # x0[:] = 0  # x0 // 2
    random_edge = random_data[x0[0] : x1[0], x0[1] : x1[1]]
    physical_edge = physical_data[x0[0] : x1[0], x0[1] : x1[1]]
    width = 0.0393701 * 190
    height = width
    fig, ax = pylab.subplots(figsize=(width, height), ncols=2, constrained_layout=True)
    vmin = min(np.min(np.abs(random_edge)), np.min(np.abs(physical_edge)))
    vmax = max(np.max(np.abs(random_edge)), np.max(np.abs(physical_edge)))
    ax[0].imshow(np.abs(physical_edge), vmin=vmin, vmax=vmax, cmap="gray_r")
    ax[1].imshow(np.abs(random_edge), vmin=vmin, vmax=vmax, cmap="gray_r")
    ax[0].set_title("Physical model", fontweight="bold")
    ax[1].set_title("Random model", fontweight="bold")
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[0].set_xlabel("(a)")
    ax[1].set_xlabel("(b)")
    fig.savefig("edge_%.1fA.png" % ps, dpi=300, bbox_inches="tight")
    pylab.close("all")
    # pylab.show()

    return np.abs(random_edge), np.abs(physical_edge)


def plot_all_mean_and_std(pixel_size, stats_list):
    """
    Make a plot of mean and std vs pixel size

    """

    # Extract the stats
    (
        p_real,
        p_imag,
        p_real_std,
        p_imag_std,
        r_real,
        r_imag,
        r_real_std,
        r_imag_std,
    ) = map(np.array, zip(*stats_list))

    width = 0.0393701 * 190
    height = width
    fig, ax = pylab.subplots(figsize=(width, height), constrained_layout=True)
    l1 = ax.plot(pixel_size, p_real, label="Physical (real)")
    l2 = ax.plot(pixel_size, p_imag, label="Physical (imag)")
    l3 = ax.plot(pixel_size, r_real, label="Random (real)")
    l4 = ax.plot(pixel_size, r_imag, label="Random (imag)")
    ax.fill_between(
        pixel_size,
        p_real - p_real_std,
        p_real + p_real_std,
        color=l1[0].get_color(),
        alpha=0.3,
    )
    ax.fill_between(
        pixel_size,
        p_imag - p_imag_std,
        p_imag + p_imag_std,
        color=l2[0].get_color(),
        alpha=0.3,
    )
    ax.fill_between(
        pixel_size,
        r_real - r_real_std,
        r_real + r_real_std,
        color=l3[0].get_color(),
        alpha=0.3,
    )
    ax.fill_between(
        pixel_size,
        r_imag - r_imag_std,
        r_imag + r_imag_std,
        color=l4[0].get_color(),
        alpha=0.3,
    )
    ax.legend(loc="lower right")
    ax.set_xlabel("Pixel size (A)")
    ax.set_ylabel("Exit wave mean and standard deviation")
    fig.savefig("mean_and_std.png", dpi=300, bbox_inches="tight")
    pylab.close("all")


def plot_all_power(pixel_size, power_list):
    """
    Plot all the power spectra

    """
    cycle = pylab.rcParams["axes.prop_cycle"].by_key()["color"]
    width = 0.0393701 * 190
    height = width
    fig, ax = pylab.subplots(figsize=(width, height), constrained_layout=True)
    for ps, power in zip(pixel_size, power_list):
        p1 = ax.plot(power[0], power[1], color=cycle[0], alpha=0.5)
        p2 = ax.plot(power[2], power[3], color=cycle[1], alpha=0.5)
    ax.set_xlabel("Spatial frequency (1/Å)")
    ax.set_ylabel("Power spectrum")
    ax.legend(handles=[p1[0], p2[0]], labels=["Physical", "Random"])
    ax.set_xlim(0, 1.0)
    # pylab.show()
    fig.savefig("power.png", dpi=300, bbox_inches="tight")
    pylab.close("all")


def plot_all_mean_and_power(pixel_size, stats_list, power_list):
    """
    Make a plot of mean and std and power vs pixel size

    """

    # Extract the stats
    (
        p_real,
        p_imag,
        p_real_std,
        p_imag_std,
        r_real,
        r_imag,
        r_real_std,
        r_imag_std,
    ) = map(np.array, zip(*stats_list))

    width = 0.0393701 * 190
    height = (3 / 8) * width
    fig, ax = pylab.subplots(figsize=(width, height), ncols=3, constrained_layout=True)
    l1 = ax[0].plot(pixel_size, p_real, label="Physical (real)")
    l2 = ax[0].plot(pixel_size, p_imag, label="Physical (imag)")
    l3 = ax[0].plot(pixel_size, r_real, label="Random (real)")
    l4 = ax[0].plot(pixel_size, r_imag, label="Random (imag)")
    ax[0].fill_between(
        pixel_size,
        p_real - p_real_std,
        p_real + p_real_std,
        color=l1[0].get_color(),
        alpha=0.3,
    )
    ax[0].fill_between(
        pixel_size,
        p_imag - p_imag_std,
        p_imag + p_imag_std,
        color=l2[0].get_color(),
        alpha=0.3,
    )
    ax[0].fill_between(
        pixel_size,
        r_real - r_real_std,
        r_real + r_real_std,
        color=l3[0].get_color(),
        alpha=0.3,
    )
    ax[0].fill_between(
        pixel_size,
        r_imag - r_imag_std,
        r_imag + r_imag_std,
        color=l4[0].get_color(),
        alpha=0.3,
    )
    ax[0].legend(loc="lower right", fontsize=6)
    ax[0].set_xlabel("Pixel size (Å)\n(a)")
    ax[0].set_ylabel("Exit wave mean")

    ax[1].scatter(pixel_size, r_real - p_real, label="Real")
    ax[1].scatter(pixel_size, r_imag - p_imag, label="Imag")
    ax[1].set_ylim((-0.01, 0.01))
    ax[1].set_yticks([-0.01, 0, 0.01])
    ax[1].legend(fontsize=8)
    ax[1].set_xlabel("Pixel size (Å)\n(b)")
    ax[1].set_ylabel("Difference in exit wave mean\n(physical - random)")
    ax[0].tick_params(axis="both", which="major", labelsize=8)
    ax[1].tick_params(axis="both", which="major", labelsize=8)
    ax[2].tick_params(axis="both", which="major", labelsize=8)
    ax[2].set(yticklabels=[])

    cycle = pylab.rcParams["axes.prop_cycle"].by_key()["color"]
    for ps, power in zip(pixel_size, power_list):
        p1 = ax[2].plot(power[0], power[1], color=cycle[0], alpha=0.5)
        p2 = ax[2].plot(power[2], power[3], color=cycle[1], alpha=0.5)
    ax[2].set_xlabel("Spatial frequency (1/Å)\n(c)")
    ax[2].set_ylabel("Power spectrum")
    ax[2].legend(handles=[p1[0], p2[0]], labels=["Physical", "Random"], fontsize=8)
    ax[2].set_xlim(0, 1.0)
    # pylab.show()
    fig.savefig("mean_and_power.png", dpi=300, bbox_inches="tight")
    pylab.close("all")


def plot_all_edge(pixel_size, edge_list):
    width = 0.0393701 * 190
    height = 0.5 * width
    fig, ax = pylab.subplots(
        figsize=(width, height), ncols=4, nrows=2, constrained_layout=True
    )

    pixel_size = [pixel_size[i] for i in [0, 3, 6, 9]]
    edge_list = [edge_list[i] for i in [0, 3, 6, 9]]

    for i, (ps, edge) in enumerate(zip(pixel_size, edge_list)):
        vmin = min(np.min(edge[0]), np.min(edge[0]))
        vmax = max(np.max(edge[1]), np.max(edge[1]))

        ax[0][i].imshow(edge[0], vmin=vmin, vmax=vmax, cmap="gray_r")
        ax[1][i].imshow(edge[1], vmin=vmin, vmax=vmax, cmap="gray_r")
        # ax[0][i].set_title("Physical model", fontweight="bold")
        # ax[1][i].set_title("Random model", fontweight="bold")
        ax[0][i].set_xticks([])
        ax[1][i].set_xticks([])
        ax[0][i].set_yticks([])
        ax[1][i].set_yticks([])
        ax[0][i].set_xlabel("(%s)" % "abcd"[i])
        ax[1][i].set_xlabel("(%s)" % "efgh"[i])
        ax[0][i].set_title("%.1f Å" % ps)

    ax[0][0].set_ylabel("Physical model")
    ax[1][0].set_ylabel("Random model")

    fig.savefig("edge.png" % ps, dpi=300, bbox_inches="tight")
    pylab.close("all")
    # pylab.show()


def validate():
    """
    Validate the ice model

    """

    # Load the water model
    atom_data = load_water_atomic_model()

    # Set the pixel sizes
    pixel_size = np.arange(0.1, 1.1, 0.1)  # [0.1]#, 1.0]

    stats_list = []
    power_list = []
    edge_list = []

    for ps in pixel_size:
        # Get the simulated exit wave
        physical_data, random_data, xmin, xmax = load_exit_wave(atom_data, ps)

        # Make the plots
        stats = plot_mean_and_var(physical_data, random_data, xmin, xmax, ps)
        power = plot_power(physical_data, random_data, xmin, xmax, ps)
        edge = plot_edge(physical_data, random_data, xmin, xmax, ps)

        stats_list.append(stats)
        power_list.append(power)
        edge_list.append(edge)

    # plot_all_mean_and_std(pixel_size, stats_list)
    # plot_all_power(pixel_size, power_list)
    plot_all_mean_and_power(pixel_size, stats_list, power_list)
    plot_all_edge(pixel_size, edge_list)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Do the ice model configuration")

    # Add arguments
    parser.add_argument(
        "-c",
        "--calibrate",
        action="store_true",
        default=False,
        dest="calibrate",
        help="Do the calibration",
    )

    parser.add_argument(
        "-v",
        "--validate",
        action="store_true",
        default=False,
        dest="validate",
        help="Do the validation",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print help
    if not args.calibrate and not args.validate:
        parser.print_help()

    # Do the calibration
    if args.calibrate:
        calibrate()

    # Do the validation
    if args.validate:
        validate()


if __name__ == "__main__":
    main()
