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
import numpy
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
    Y, X = numpy.indices((data.shape))
    ysize, xsize = data.shape
    distance = numpy.sqrt((X - xsize / 2.0) ** 2 + (Y - ysize / 2.0) ** 2)
    distance = numpy.floor(distance).astype("int32")
    c1 = numpy.bincount(distance.ravel()).astype("float32")
    c2 = numpy.bincount(distance.ravel(), data.ravel()).astype("float32")
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

    return input_multislice


def compute_projected_potential():
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
            V = numpy.array(V)
            potential.append(V)

        # Run the simulation
        multem.compute_projected_potential(system_conf, input_multislice, callback)

        # Save the potential
        potential = numpy.sum(potential, axis=0)
        filename = "projected_potential_%.1f_%d.npz" % (pixel_size, thickness)
        numpy.savez(filename, potential=potential, num_atoms=num_atoms)

    # Read the atom data
    atom_data = load_water_atomic_model()

    # Simulate the projected potential
    for pixel_size in numpy.arange(0.1, 2.1, 0.1):
        for thickness in numpy.arange(5, 25, 5):
            print(
                "Compute projected potential for px = %f, dz = %f"
                % (pixel_size, thickness)
            )
            if not os.path.exists(
                "projected_potential_%.1f_%d.npz" % (pixel_size, thickness)
            ):
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
        x0 = numpy.random.uniform(0, 1) + nx // 2
        y0 = numpy.random.uniform(0, 1) + ny // 2
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
            V = numpy.array(V)
            thickness.append(z1 - z0)
            potential.append(V)

        # Run the simulation
        multem.compute_projected_potential(system_conf, input_multislice, callback)

        # Compute the mean potential
        V = numpy.sum(potential, axis=0)
        means.append(numpy.mean(V))

    # Return the size and mean potential
    return size, numpy.mean(means)


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
    Compute the variance correction table

    """

    size = 40

    pixel_size = numpy.arange(0.1, 2.1, 0.1)
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
    ax.plot(area, mean)
    ax.set_xlabel(r"Pixel area ($Å^2$)" + "\n(b)")
    ax.set_title("Mean correction factor")


def compute_variance_correction(ax=None):
    """
    Compute the variance correction table

    """

    X = []
    Y = []

    # Loop through the pixel sizes
    thickness = 20
    for pixel_size in numpy.arange(0.1, 2.1, 0.1):

        # Read the projected potential
        handle = numpy.load("projected_potential_%.1f_%d.npz" % (pixel_size, thickness))
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
        var = numpy.var(potential)
        density = num_molecules / (area)

        # Append the pixel area and variance / density
        X.append(pixel_size**2)
        Y.append(var / density)

    X = numpy.array(X)
    Y = numpy.array(Y)

    # Extrapolate to zero and nornalize
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
    ax.plot(X, Y)
    ax.set_xlabel(r"Pixel area ($Å^2$)" + "\n(c)")
    ax.set_title("Variance correction factor")


def compute_power(ax=None):
    """
    Compute the power spectrum

    """

    # Compute the fit to the power spectrum
    pixel_size = 0.1
    for thickness in [20]:  # , 19, 18, 15, 10, 5]:

        # Read the projected potential
        handle = numpy.load("projected_potential_%.1f_%d.npz" % (pixel_size, thickness))
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
        area = (nx * pixel_size) * (ny * pixel_size)
        var = numpy.var(potential)
        mean = numpy.mean(potential)
        # density = num_molecules / (area)
        potential -= mean
        print("Mean: %.3f; Variance: %.3f" % (mean, var))

        # Compute the FFT of the data and the power spectrum
        fft_data = numpy.fft.fft2(potential)
        power = numpy.abs(fft_data) ** 2
        Y, X = numpy.mgrid[0 : power.shape[0], 0 : power.shape[1]]
        q = (1 / pixel_size) * numpy.sqrt(
            ((X - power.shape[1] / 2) / power.shape[1]) ** 2
            + ((Y - power.shape[0] / 2) / power.shape[0]) ** 2
        )
        q = numpy.fft.fftshift(q)

        def func(q, A0, A1, A2, A3):
            M = 1.0 / 2.88
            P = A0 * numpy.exp(-0.5 * q**2 / A1**2) + A2 * numpy.exp(
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
        #     A0 = numpy.abs(A0)
        #     A2 = numpy.abs(A2)
        #     # A0 = 0.1608
        #     # A2, A3 = 0.822372155, 0.08153797

        #     # A1, A3 = 0.68518427, 0.08693241
        #     M = 1.0 / 2.88
        #     P = A0 * numpy.exp(-0.5 * q ** 2 / A1 ** 2) + A2 * numpy.exp(
        #         -0.5 * (q - M) ** 2 / A3 ** 2
        #     )
        #     # I = (2*pi*(A0*A1**2 + A2*M*sqrt(2*pi*A3**2)))
        #     # P /= I
        #     model = P
        #     model[0, 0] = normalized_power[0, 0]
        #     W = 1.0 / (q * power.shape[0] + 1)
        #     A = numpy.sum(W * (model - normalized_power) ** 2) / numpy.sum(W)
        #     B = (numpy.sum(model) - numpy.sum(normalized_power)) ** 2 / model.size
        #     print(p, A, B)
        #     return A + B

        # Compute the variance correction factor
        Cv = numpy.exp(-3.2056 * pixel_size**2)
        C = Cv * num_molecules / (pixel_size**4)

        # Compute the total integral of the power
        I = numpy.sum(power) * (1 / pixel_size) ** 2 / power.size
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
        rp = radial_average(numpy.fft.fftshift(power))
        rm = radial_average(numpy.fft.fftshift(model))
        d = numpy.arange(rp.size) / (pixel_size * power.shape[0])

        # Plot the power spectrum and best fit
        ax.plot(d[1:], rp[1:] / C, label="%d" % thickness)
        ax.plot(d[1:], rm[1:] / C, color="black", alpha=0.5, label="Model")

    # Set some plot properties
    ax.set_xlabel("Spatial frequency ($Å$)\n(a)")
    ax.set_title("Power spectrum")
    ax.set_xlim(0, 1)
    ax.set_yticklabels("")


def calibrate():
    """
    Calibrate ice model

    """
    # Get the water atomic model file
    get_water_atomic_model()

    # Compute the projected potential
    compute_projected_potential()

    # Setup the figure
    width = 0.0393701 * 190
    height = width / 3.0
    fig, ax = pylab.subplots(ncols=3, figsize=(width, height), constrained_layout=True)
    compute_power(ax[0])
    compute_mean_correction(ax[1])
    compute_variance_correction(ax[2])
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
    z_min = atom_data.data["z"].min()
    z_max = atom_data.data["z"].max()
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
    nx = next_power_2(int(x_box_size / pixel_size))
    ny = next_power_2(int(y_box_size / pixel_size))
    assert nx <= 4096
    assert ny <= 4096
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
    input_multislice.spec_dz = 5

    # Set the specimen atoms
    input_multislice.spec_atoms = atom_data.to_multem()

    # Run the simulation
    output_multislice = multem.simulate(system_conf, input_multislice)

    # Get the image
    physical_image = numpy.array(output_multislice.data[0].psi_coh).T

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
    random_image = numpy.array(output_multislice.data[0].psi_coh).T

    # Return the images
    x0 = numpy.array((x_box_size / 2 - x_size / 2, y_box_size / 2 - y_size / 2))
    x1 = numpy.array((x_box_size / 2 + x_size / 2, y_box_size / 2 + y_size / 2))
    return physical_image, random_image, x0, x1


def validate():
    """
    Validate the ice model

    """

    # Load the water model
    atom_data = load_water_atomic_model()

    pixel_size = [1.0, 0.1]

    for ps in pixel_size:

        # Get the simulated exit wave
        physical_data, random_data, xmin, xmax = compute_exit_wave(atom_data, ps)

        x0 = numpy.floor(xmin / ps).astype("int32")
        x1 = numpy.floor(xmax / ps).astype("int32")
        xr = x1 - x0
        x0 = x0 + xr // 4
        x1 = x1 - xr // 4

        random_middle = random_data[x0[0] : x1[0], x0[1] : x1[1]]
        physical_middle = physical_data[x0[0] : x1[0], x0[1] : x1[1]]
        physical_middle_mean_real = numpy.mean(physical_middle.flatten().real)
        physical_middle_mean_imag = numpy.mean(physical_middle.flatten().imag)
        random_middle_mean_real = numpy.mean(random_middle.flatten().real)
        random_middle_mean_imag = numpy.mean(random_middle.flatten().imag)

        # pylab.imshow(numpy.abs(random_middle))
        # pylab.show()
        # pylab.imshow(numpy.abs(physical_middle))
        # pylab.show()
        # continue

        physical_middle_std_real = numpy.std(physical_middle.flatten().real)
        physical_middle_std_imag = numpy.std(physical_middle.flatten().imag)
        random_middle_std_real = numpy.std(random_middle.flatten().real)
        random_middle_std_imag = numpy.std(random_middle.flatten().imag)

        # print("Hola")
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
            "mean: %.2f\n sdev: %.2f"
            % (random_middle_mean_real, random_middle_std_real),
        )
        ax[1][1].text(
            xi,
            0.5 * ax[1][1].get_ylim()[1],
            "mean: %.2f\n sdev: %.2f"
            % (random_middle_mean_imag, random_middle_std_imag),
        )
        fig.savefig("histograms_%.1fA.png" % ps, dpi=300, bbox_inches="tight")

        def compute_power(data, pixel_size):
            f = numpy.fft.fft2(data)
            p = numpy.abs(f) ** 2
            p = numpy.fft.fftshift(p)

            r = radial_average(p)[0 : data.shape[0] // 2]
            d = numpy.arange(r.size) / (pixel_size * data.shape[0])
            return d[1:], r[1:]

        random_d, random_power = compute_power(random_middle, ps)
        physical_d, physical_power = compute_power(physical_middle, ps)

        width = 0.0393701 * 190
        height = width * 0.74
        fig, ax = pylab.subplots(figsize=(width, height), constrained_layout=True)
        ax.plot(physical_d, physical_power, label="Physical model")
        ax.plot(random_d, random_power, label="Random model")
        ax.set_xlabel("Spatial frequency (1/Å)")
        ax.set_ylabel("Power spectrum")
        ax.set_xlim(0, 0.5)
        ax.legend()
        fig.savefig("power_%.1fA.png" % ps, dpi=300, bbox_inches="tight")

        x0 = numpy.floor(xmin / ps).astype("int32")
        # x1 = numpy.floor(xmax / ps).astype("int32")
        x1 = 2 * x0  # + x0 // 2
        x0[:] = 0  # x0 // 2
        random_edge = random_data[x0[0] : x1[0], x0[1] : x1[1]]
        physical_edge = physical_data[x0[0] : x1[0], x0[1] : x1[1]]
        width = 0.0393701 * 190
        height = width
        fig, ax = pylab.subplots(
            figsize=(width, height), ncols=2, constrained_layout=True
        )
        vmin = min(
            numpy.min(numpy.abs(random_edge)), numpy.min(numpy.abs(physical_edge))
        )
        vmax = max(
            numpy.max(numpy.abs(random_edge)), numpy.max(numpy.abs(physical_edge))
        )
        ax[0].imshow(numpy.abs(physical_edge), vmin=vmin, vmax=vmax)
        ax[1].imshow(numpy.abs(random_edge), vmin=vmin, vmax=vmax)
        ax[0].set_title("Physical model", fontweight="bold")
        ax[1].set_title("Random model", fontweight="bold")
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[0].set_xlabel("(a)")
        ax[1].set_xlabel("(b)")
        fig.savefig("edge_%.1fA.png" % ps, dpi=300, bbox_inches="tight")
        # pylab.show()


if __name__ == "__main__":

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
