<div align="center">

![Parakeet :parrot:](docs/source/images/parakeet_logo.png)

> **Parakeet** is a digital twin for cryo electron tomography and stands for **P**rogram for **A**nalysis and **R**econstruction of **A**rtificial data for **K**ryo **E**l**E**ctron **T**omography

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![PyPI version shields.io](https://img.shields.io/pypi/v/python-parakeet.svg)](https://pypi.python.org/pypi/python-parakeet/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/python-parakeet.svg)](https://pypi.python.org/pypi/python-parakeet/)
[![PyPI download month](https://img.shields.io/pypi/dm/python-parakeet.svg)](https://pypi.python.org/pypi/python-parakeet/)

[![Building](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/python-package.yml/badge.svg)](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/python-package.yml)
[![Publishing](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/python-publish.yml)

[![Docker](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/docker-publish.yml)
[![Quay IO](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/quay-io-publish.yml/badge.svg)](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/quay-io-publish.yml)
[![Conda](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/conda.yml/badge.svg)](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/conda.yml)
[![Snapcraft](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/snapcraft.yml/badge.svg)](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/snapcraft.yml)

[![Documentation](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/sphinx.yml/badge.svg)](https://github.com/rosalindfranklininstitute/parakeet/actions/workflows/sphinx.yml)
[![DOI](https://zenodo.org/badge/204956111.svg)](https://zenodo.org/badge/latestdoi/204956111)

</div>

## Installation

Parakeet can be installed using pip with the following command:

```sh
  pip install python-parakeet
```

> **_NOTE:_** Because the package needs to be built locally from source and has
some external dependencies you may need to ensure your environment is ready before
running this command. You need to set the location of the CUDA compiler and
possibly G++ and FFTW libraries. For full instructions please see the
installation documentation
[here](https://rosalindfranklininstitute.github.io/parakeet/installation.html).

## Usage

Parakeet can be used as a suite of command line tools as follows:

```sh
  parakeet.config.new -c config.yaml
  parakeet.sample.new -c config.yaml
  parakeet.sample.add_molecules -c config.yaml
  parakeet.simulate.exit_wave -c config.yaml
  parakeet.simulate.optics -c config.yaml
  parakeet.simulate.image -c config.yaml
```

For full command line usage instructions please see the command line
documentation here
[here](https://rosalindfranklininstitute.github.io/parakeet/usage.html).
Alternatively, there is a complementary high level Python API which can be seen
[here](https://rosalindfranklininstitute.github.io/parakeet/api.html).

## Documentation

Checkout the [documentation](https://rosalindfranklininstitute.github.io/parakeet/) for more information!

## Notifications

You can receive notifications from the [Github discussions](https://github.com/rosalindfranklininstitute/parakeet/discussions)
by clicking "watch" on this repository.

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/parakeet/issues) to submit bugs or request features.

## Changes

Checkout the [changelog](CHANGES.rst) for a list of recent changes to parakeet.

## Contributions

If you would like to help contribute to profet, please read our [contribution](CONTRIBUTING.md) guide and [code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright Diamond Light Source and Rosalind Franklin Institute, 2019.

Distributed under the terms of the GPLv3 license, parakeet is free and open source software.

