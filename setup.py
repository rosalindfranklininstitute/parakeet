#
# Copyright (C) 2019 James Parkhurst
#
# This code is distributed under the BSD license.
#
from skbuild import setup


def main():
    """
    Setup the package

    """
    tests_require = ["pytest", "pytest-cov", "mock"]

    setup(
        package_dir={"": "src"},
        packages=["elfantasma"],
        install_requires=[
            "dask",
            "distributed",
            "dask_jobqueue",
            "gemmi",
            "h5py",
            "mrcfile",
            "numpy",
            "pandas",
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
                "elfantasma-show-config=elfantasma.command_line:show_config_main",
                "elfantasma-create-sample=elfantasma.command_line:create_sample",
                "elfantasma-freeze=elfantasma.command_line:freeze",
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
