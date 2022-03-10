# parakeet
> **Parakeet** is a digital twin for cryo electron tomography and stands for **P**rogram for **A**nalysis and **R**econstruction of **A**rtificial data for **K**ryo **E**l**E**ctron **T**omography

![Parakeet](docs/source/images/parakeet_small.png)

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/rosalindfranklininstitute/amplus-digital-twin.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/amplus-digital-twin/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/rosalindfranklininstitute/amplus-digital-twin.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/amplus-digital-twin/alerts/)
[![Building](https://github.com/rosalindfranklininstitute/amplus-digital-twin/actions/workflows/python-package.yml/badge.svg)](https://github.com/rosalindfranklininstitute/amplus-digital-twin/actions/workflows/python-package.yml)
[![Publishing](https://github.com/rosalindfranklininstitute/amplus-digital-twin/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rosalindfranklininstitute/amplus-digital-twin/actions/workflows/python-publish.yml)
[![DOI](https://zenodo.org/badge/204956111.svg)](https://zenodo.org/badge/latestdoi/204956111)


## Installation

Parakeet can be installed using pip with the following command:

```sh
  pip install python-parakeet
```

However, because the package needs to be built locally from source and has some
external dependencies you may need to ensure your environment is ready before
running this command. For full instructions please see the installation
documentation
[here](https://rosalindfranklininstitute.github.io/amplus-digital-twin/installation.html).

## Usage

Parakeet can be used as a suite of command line tools as follows:

```sh
  parakeet.sample.new -c config.yaml
  parakeet.sample.add_molecules -c config.yaml
  parakeet.simulate.exit_wave -c config.yaml
  parakeet.simulate.optics -c config.yaml
  parakeet.simulate.image -c config.yaml
```

For full command line usage instructions please see the command line
documentation here
[here](https://rosalindfranklininstitute.github.io/amplus-digital-twin/usage.html).
Alternatively, there is a complementary high level Python API which can be seen
[here](https://rosalindfranklininstitute.github.io/amplus-digital-twin/api.html).

## Documentation

Checkout the [documentation](https://rosalindfranklininstitute.github.io/amplus-digital-twin/) for more information!

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/parakeet/issues) to submit bugs or request features.

## License

Copyright Diamond Light Source and Rosalind Franklin Institute, 2019.

Distributed under the terms of the GPLv3 license, parakeet is free and open source software.

