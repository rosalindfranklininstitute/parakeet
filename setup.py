#
# Copyright (C) 2019 James Parkhurst
#
# This code is distributed under the BSD license.
#
import os
import subprocess
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """
    Build the extensions

    """

    def build_extensions(self):

        # Set the cmake directory
        cmake_lists_dir = os.path.abspath(".")

        # Ensure the build directory exists
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run Cmake once
        ext = self.extensions[0]

        # Get the directory
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Arguments to cmake
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=%s" % "Release",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%s" % extdir,
            "-DPYTHON_EXECUTABLE=%s" % sys.executable,
        ]

        # Config and the extension
        subprocess.check_call(
            ["cmake", cmake_lists_dir] + cmake_args, cwd=self.build_temp
        )

        # Build the extension
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)


def main():
    """
    Setup the package

    """
    tests_require = ["pytest", "pytest-cov", "mock"]

    setup(
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        setup_requires=["dask", "pytest-runner"],
        install_requires=[
            "distributed",
            "dask_jobqueue",
            "gemmi",
            "guanaco @ git+https://github.com/rosalindfranklininstitute/guanaco.git@master#egg=guanaco",
            "h5py",
            "maptools @ git+https://github.com/rosalindfranklininstitute/maptools.git@master#egg=maptools",
            "mrcfile",
            "numpy",
            "pandas",
            "pillow",
            "python-multem @ git+https://github.com/rosalindfranklininstitute/python-multem.git@master#egg=python-multem",
            "scipy",
            "pyyaml",
        ],
        tests_require=tests_require,
        test_suite="tests",
        ext_modules=[Extension("amplus_ext", [])],
        cmdclass={"build_ext": CMakeBuild},
        entry_points={
            "console_scripts": [
                "amplus.read_pdb=amplus.command_line:read_pdb",
                "amplus.export=amplus.command_line:export",
                "amplus.config.show=amplus.command_line.config:show",
                "amplus.config.edit=amplus.command_line.config:edit",
                "amplus.sample.new=amplus.command_line.sample:new",
                "amplus.sample.add_molecules=amplus.command_line.sample:add_molecules",
                "amplus.sample.mill=amplus.command_line.sample:mill",
                "amplus.sample.sputter=amplus.command_line.sample:sputter",
                "amplus.sample.show=amplus.command_line.sample:show",
                "amplus.simulate.projected_potential=amplus.command_line.simulate:projected_potential",
                "amplus.simulate.exit_wave=amplus.command_line.simulate:exit_wave",
                "amplus.simulate.optics=amplus.command_line.simulate:optics",
                "amplus.simulate.image=amplus.command_line.simulate:image",
                "amplus.simulate.simple=amplus.command_line.simulate:simple",
                "amplus.simulate.ctf=amplus.command_line.simulate:ctf",
                "amplus.analyse.reconstruct=amplus.command_line.analyse:reconstruct",
                "amplus.analyse.average_particles=amplus.command_line.analyse:average_particles",
                "amplus.analyse.refine=amplus.command_line.analyse:refine",
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
