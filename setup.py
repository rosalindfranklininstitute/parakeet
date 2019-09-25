#
# Copyright (C) 2019 James Parkhurst
#
# This code is distributed under the BSD license.
#
from setuptools import setup
from setuptools import find_packages


def main():
    """
    Setup the package

    """
    tests_require = ["pytest", "pytest-cov", "mock"]

    setup(
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        install_requires=[
            "dask",
            "distributed",
            "dask_jobqueue",
            "gemmi",
            "h5py",
            "mrcfile",
            "numpy",
            "pillow",
            "python-multem",
            "scipy",
            "pyyaml",
        ],
        setup_requires=["pytest-runner"],
        tests_require=tests_require,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "elfantasma=elfantasma.command_line:main",
                "elfantasma-read-pdb=elfantasma.command_line:read_pdb",
                "elfantasma-convert=elfantasma.command_line:convert",
            ]
        },
        extras_require={
            "build_sphinx": ["sphinx", "sphinx_rtd_theme"],
            "test": tests_require,
        },
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
