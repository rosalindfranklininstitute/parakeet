#
# amplus.util.compute_ice_noise_filter.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#

import amplus.io
from selknam.viewer.widgets.radial_average_plot import radial_average
from matplotlib import pylab
import numpy
import scipy.optimize
from math import pi


def compute_ice_noise_filter_parameters(images, pixel_size, plot=False):
    """
    Given a stack of images, compute the power spectrum parameters

    """

    # We should have a stack on images
    assert len(images.shape) == 3

    # Compute the mean power spectrum
    mean_power = numpy.zeros(shape=images.shape[1:], dtype="float64")
    power = numpy.zeros(dtype="float64", shape=images.shape)
    for i in range(images.shape[0]):

        # Normalize the image
        data = images[i, :, :]
        mean = numpy.mean(data)
        sdev = numpy.std(data)
        data = (data - mean) / sdev

        # Compute the power spectrum
        fft_data = numpy.fft.fft2(data)
        power[i, :, :] = numpy.abs(fft_data) ** 2 / data.size
        mean_power += power[i, :, :]
    mean_power /= images.shape[0]

    # Compute the radial average of the power spectrum
    size = min((mean_power.shape[0] // 2, mean_power.shape[1] // 2))
    radial_power = radial_average(numpy.fft.fftshift(mean_power))[0:size]

    # The function to fit
    def func(f, a, b, s):
        return a * numpy.exp(-0.5 * (f / s) ** 2) + b

    # Fit the function to the radial power spectrum
    a = numpy.mean(radial_power)
    b = numpy.mean(radial_power)
    s = 0.3
    f = (1.0 / pixel_size) * numpy.arange(len(radial_power)) / len(radial_power)
    (a, b, s), pcov = scipy.optimize.curve_fit(
        func, f[1:], radial_power[1:], p0=(a, b, s)
    )

    # The parameters
    s = abs(s)
    # print(a, b, s)
    # a= 2.641625
    # b= 0.288581
    # s= 0.425859

    # Plot some stuff
    if plot:

        # Plot each radial average in turn
        fig, ax = pylab.subplots()
        x = numpy.arange(len(radial_power)) / len(radial_power)
        for i in range(images.shape[0]):
            r = radial_average(numpy.fft.fftshift(power[i]))[0:size]
            ax.plot(f, r)
        ax.plot(f, func(f, a, b, s), color="black")
        ax.set_xlabel("Frequency (1/A)")
        ax.set_ylabel("Power")
        ax.set_title("Mean radial average power spectrum")
        pylab.show()

        # Plot the mean and the fit
        fig, ax = pylab.subplots()
        x = numpy.arange(len(radial_power)) / len(radial_power)
        ax.plot(f, radial_power)
        ax.plot(f, func(f, a, b, s), color="black")
        ax.set_xlabel("Frequency (1/A)")
        ax.set_ylabel("Power")
        ax.set_title("Mean radial average power spectrum")
        pylab.show()

    # Return the parameters
    return a, b, s


def compute_power_spectrum(size, pixel_size, a, b, s):
    """
    Compute the power spectrum from the parameters

    """

    # Generate the indices and the distance from the centre
    X, Y = numpy.indices(size)
    f = numpy.sqrt(
        (X - size[0] / 2.0) ** 2 / size[0] ** 2
        + (Y - size[1] / 2.0) ** 2 / size[1] ** 2
    ) * (1.0 / pixel_size)

    # Compute the power spectrum
    return numpy.fft.fftshift(numpy.exp(-0.5 * (f / s) ** 2) * a + b)


def generate_noise(size, pixel_size, a, b, s, plot=False):
    """
    Generate the Fourier filtered noise

    """
    # Compute the power spectrum
    power_spectrum = compute_power_spectrum(size, pixel_size, *params)

    # Plot the power spectrum
    if plot:
        pylab.imshow(power_spectrum)
        pylab.show()

    # The amplitude and phase
    amplitude = numpy.sqrt(power_spectrum)
    # noise = numpy.random.normal(0,1,size=amplitude.shape) + 1j * numpy.random.normal(0,1,size=amplitude.shape)
    # fft_data = numpy.fft.fft2(noise)
    phase = numpy.random.uniform(-pi, pi, size=power_spectrum.shape)
    fft_data = amplitude * numpy.exp(1j * phase)

    # Return the complex noise image
    return numpy.fft.ifft2(fft_data)


def generate_ice_noise(template, pixel_size, a, b, s):

    # Generate the noise
    size = template.shape
    data = generate_noise(size, 2, *params)

    x_real = numpy.real(x)
    x_imag = numpy.imag(x)
    y_real = numpy.real(y)
    y_imag = numpy.imag(y)

    # Compute the covariance matrix of the template
    y_cov = numpy.cov(
        y_real.flatten().astype("float64"), y_imag.flatten().astype("float64")
    )
    C = numpy.linalg.cholesky(y_cov)

    mx = numpy.mean(x_real)
    sx = numpy.std(x_real)
    x_real = (x_real - mx) / sx

    mx = numpy.mean(x_imag)
    sx = numpy.std(x_imag)
    x_imag = (x_imag - mx) / sx

    data = numpy.matmul(C, numpy.array([x_real.flatten(), x_imag.flatten()]))
    x_cov = numpy.cov(data[0, :], data[1, :])
    data = (data[0, :] + 1j * data[1, :]).reshape(x.shape)
    return data + numpy.mean(y)


if __name__ == "__main__":

    def rebin(a, shape):
        sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    mr = []
    mi = []
    sr = []
    si = []
    ab = []
    cc = []
    thickness = []
    images = []
    for thick in range(500, 10001, 500):
        handle = amplus.io.open(
            "/home/upc86896/Desktop/2020-02-17-only-ice-different-thickness-large/%d_exit_wave.h5"
            % thick
        )
        data = handle.data[0, 250:750, 250:750]
        thickness.append(thick)
        mr.append(numpy.mean(numpy.real(data)))
        mi.append(numpy.mean(numpy.imag(data)))
        sr.append(numpy.std(numpy.real(data)))
        si.append(numpy.std(numpy.imag(data)))
        ab.append(numpy.mean(numpy.abs(data)))
        cc.append(
            numpy.corrcoef(numpy.real(data).flatten(), numpy.imag(data).flatten())[0, 1]
        )

        print(
            numpy.mean(data), numpy.std(numpy.real(data)), numpy.std(numpy.imag(data))
        )
        # data = rebin(data, (250, 250))
        images.append(data)
    pylab.plot(thickness, mr, label="Mean (real)")
    pylab.plot(thickness, mi, label="Mean (imag)")
    pylab.plot(thickness, sr, label="Sdev (real)")
    pylab.plot(thickness, si, label="Sdev (imag)")
    pylab.plot(thickness, ab, label="Mean (abs)")
    pylab.plot(thickness, cc, label="CC (real,imag)")
    pylab.legend()
    pylab.show()

    pixel_size = 2

    # Compute the fit to the power spectrum
    params = compute_ice_noise_filter_parameters(
        numpy.array(images), pixel_size, plot=True
    )

    # Print the parameters
    print("Parameters:")
    print("  A: %f" % params[0])
    print("  B: %f" % params[1])
    print("  S: %f" % params[2])

    size = images[0].shape
    noise = generate_noise(size, pixel_size, *params)

    for i in range(len(images)):

        def normalize_complex(x, y):
            x_real = numpy.real(x)
            x_imag = numpy.imag(x)
            y_real = numpy.real(y)
            y_imag = numpy.imag(y)

            y_cov = numpy.cov(
                y_real.flatten().astype("float64"), y_imag.flatten().astype("float64")
            )
            C = numpy.linalg.cholesky(y_cov)

            mx = numpy.mean(x_real)
            sx = numpy.std(x_real)
            x_real = (x_real - mx) / sx

            mx = numpy.mean(x_imag)
            sx = numpy.std(x_imag)
            x_imag = (x_imag - mx) / sx
            # x_real = numpy.random.normal(0, 1, size=x_real.shape)
            # x_imag = numpy.random.normal(0, 1, size=x_real.shape)

            # print(numpy.std(x_real), numpy.std(x_imag))
            data = numpy.matmul(C, numpy.array([x_real.flatten(), x_imag.flatten()]))
            # x_cov = numpy.cov(x_real.flatten(), x_imag.flatten())
            x_cov = numpy.cov(data[0, :], data[1, :])
            data = (data[0, :] + 1j * data[1, :]).reshape(x.shape)
            # data = x_real + 1j*x_imag
            print(y_cov)
            print(x_cov)
            return data + numpy.mean(y)
            # return normalize(x_real, y_real) + 1j * normalize(x_imag, y_imag)

        image = images[i]
        data = normalize_complex(noise, image)

        # m_image = numpy.mean(image)
        # cov_image = numpy.cov(numpy.real(image).flatten(), numpy.imag(image).flatten())
        # m_data = numpy.mean(data)
        # cov_data = numpy.cov(numpy.real(data).flatten(), numpy.imag(data).flatten())
        # c_image = sqrt(cov_image[0,0]) + 1j*sqrt(cov_image[1,1])
        # c_data = sqrt(cov_data[0,0]) + 1j* sqrt(cov_data[1,1])
        # # c_image =  1j*sqrt(cov_image[1,1])
        # # c_data =  1j* sqrt(cov_data[1,1])
        # print(c_image / c_data)

        # data = (data - m_data) * c_image / c_data + m_image

        # sr, si = numpy.std(numpy.real(image)), numpy.std(numpy.imag(image))
        # data = (data-numpy.mean(data))*s/numpy.std(data) + m

        cc_image = numpy.corrcoef(
            numpy.real(image).flatten(), numpy.imag(image).flatten()
        )
        cc_data = numpy.corrcoef(numpy.real(data).flatten(), numpy.imag(data).flatten())

        # fig, ax = pylab.subplots(ncols=2)
        # ax[0].plot(numpy.real(image)[250,:])
        # ax[1].plot(numpy.real(data)[250,:])
        # pylab.show()

        print("ALL")
        print("M: ", numpy.mean(data), numpy.mean(image))
        print("S: ", numpy.std(data), numpy.std(image))
        print("C: ", cc_image[0][1], cc_data[0][1])

        # fft_data = numpy.fft.fft2(image)
        # fft_data = amplitude * numpy.exp(1j*numpy.angle(fft_data))
        # data = numpy.fft.ifft2(fft_data)
        # data = (data-numpy.mean(data))*s/numpy.std(data) + m

        # noise = numpy.random.normal(0, 1, size=fft_data.shape)
        # fft_data = numpy.fft.fft2(noise)
        # fft_data = amplitude * fft_data#)*numpy.exp(1j*numpy.angle(fft_data))
        # data = numpy.fft.ifft2(fft_data)

        # fig, ax = pylab.subplots(ncols=2)
        # ax[0].imshow(numpy.abs(numpy.fft.fft(numpy.angle(fft_data)))**2)
        # ax[1].imshow(numpy.abs(numpy.fft.fft(phase))**2)
        # pylab.show()

        # fig, ax = pylab.subplots(ncols=2)
        # ax[0].hist(numpy.real(image).flatten())
        # ax[1].hist(numpy.real(data).flatten())
        # pylab.show()

        print("REAL")
        a = numpy.real(image)
        b = numpy.real(data)
        print("M: ", numpy.mean(a), numpy.mean(b))
        print("S: ", numpy.std(a), numpy.std(b))
        print("Min: ", numpy.min(a), numpy.min(b))
        print("Max: ", numpy.max(a), numpy.max(b))
        vmin = numpy.min(a)
        vmax = numpy.max(a)
        fig, ax = pylab.subplots(ncols=2)
        ax[0].imshow(a, vmin=vmin, vmax=vmax)
        ax[1].imshow(b, vmin=vmin, vmax=vmax)
        pylab.show()

        print("IMAG")
        a = numpy.imag(image)
        b = numpy.imag(data)
        print("M: ", numpy.mean(a), numpy.mean(b))
        print("S: ", numpy.std(a), numpy.std(b))
        print("Min: ", numpy.min(a), numpy.min(b))
        print("Max: ", numpy.max(a), numpy.max(b))
        vmin = numpy.min(a)
        vmax = numpy.max(a)
        fig, ax = pylab.subplots(ncols=2)
        ax[0].imshow(a, vmin=vmin, vmax=vmax)
        ax[1].imshow(b, vmin=vmin, vmax=vmax)
        pylab.show()

        print("ABS")
        a = numpy.abs(image)
        b = numpy.abs(data)
        print("M: ", numpy.mean(a), numpy.mean(b))
        print("S: ", numpy.std(a), numpy.std(b))
        print("Min: ", numpy.min(a), numpy.min(b))
        print("Max: ", numpy.max(a), numpy.max(b))
        vmin = numpy.min(a)
        vmax = numpy.max(a)
        fig, ax = pylab.subplots(ncols=2)
        ax[0].imshow(a, vmin=vmin, vmax=vmax)
        ax[1].imshow(b, vmin=vmin, vmax=vmax)
        pylab.show()

        print("ABS**2")
        a = numpy.abs(image) ** 2
        b = numpy.abs(data) ** 2
        print("M: ", numpy.mean(a), numpy.mean(b))
        print("S: ", numpy.std(a), numpy.std(b))
        print("Min: ", numpy.min(a), numpy.min(b))
        print("Max: ", numpy.max(a), numpy.max(b))
        vmin = numpy.min(a)
        vmax = numpy.max(a)
        fig, ax = pylab.subplots(ncols=2)
        ax[0].imshow(a, vmin=vmin, vmax=vmax)
        ax[1].imshow(b, vmin=vmin, vmax=vmax)
        pylab.show()

        # image2 = image[100:200, 100:200]
        # data2 = data[100:200, 100:200]
        # vmin = numpy.min(numpy.real(image2))**2
        # vmax = numpy.max(numpy.real(image2))**2
        # fig, ax = pylab.subplots(ncols=2)
        # ax[0].imshow(numpy.real(image2)**2, vmin=vmin, vmax=vmax)
        # ax[1].imshow(numpy.real(data2)**2, vmin=vmin, vmax=vmax)
        # pylab.show()
