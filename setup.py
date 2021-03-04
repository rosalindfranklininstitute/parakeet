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
        packages=["amplus"],
        setup_requires=["dask", "pytest-runner"],
        install_requires=[
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
        tests_require=tests_require,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "amplus.read_pdb=amplus.command_line:read_pdb",
                "amplus.show_config=amplus.command_line:show_config_main",
                "amplus.export=amplus.command_line:export",
                "amplus.sample.new=amplus.command_line.sample:new",
                "amplus.sample.add_molecules=amplus.command_line.sample:add_molecules",
                "amplus.sample.mill=amplus.command_line.sample:mill",
                "amplus.sample.show=amplus.command_line.sample:show",
                "amplus.simulate.projected_potential=amplus.command_line.simulate:projected_potential",
                "amplus.simulate.exit_wave=amplus.command_line.simulate:exit_wave",
                "amplus.simulate.optics=amplus.command_line.simulate:optics",
                "amplus.simulate.image=amplus.command_line.simulate:image",
                "amplus.simulate.simple=amplus.command_line.simulate:simple",
                "amplus.simulate.ctf=amplus.command_line.simulate:ctf",
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
