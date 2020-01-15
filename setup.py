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
                "elfantasma.read_pdb=elfantasma.command_line:read_pdb",
                "elfantasma.show_config=elfantasma.command_line:show_config_main",
                "elfantasma.export=elfantasma.command_line:export",
                "elfantasma.sample.new=elfantasma.command_line.sample:new",
                "elfantasma.sample.add_molecules=elfantasma.command_line.sample:add_molecules",
                "elfantasma.sample.mill=elfantasma.command_line.sample:mill",
                "elfantasma.sample.show=elfantasma.command_line.sample:show",
                "elfantasma.simulate.exit_wave=elfantasma.command_line.simulate:exit_wave",
                "elfantasma.simulate.optics=elfantasma.command_line.simulate:optics",
                "elfantasma.simulate.image=elfantasma.command_line.simulate:image",
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
