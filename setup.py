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
        install_requires=[
            "distributed",
            "dask_jobqueue",
            "gemmi",
            "guanaco",
            "h5py",
            "maptools",
            "mrcfile",
            "numpy",
            "pandas",
            "pillow",
            "profet",
            "pydantic",
            "python-multem",
            "pyyaml",
            "scipy",
            "starfile",
            "lxml_html_clean",
        ],
        tests_require=tests_require,
        test_suite="tests",
        ext_modules=[Extension("parakeet_ext", [])],
        cmdclass={"build_ext": CMakeBuild},
        use_scm_version={"write_to": "src/parakeet/_version.py"},
        entry_points={
            "console_scripts": [
                "parakeet=parakeet.command_line:main",
                "parakeet.export=parakeet.command_line:export",
                "parakeet.run=parakeet.command_line:run",
                "parakeet.pdb.get=parakeet.command_line.pdb:get",
                "parakeet.pdb.read=parakeet.command_line.pdb:read",
                "parakeet.config.show=parakeet.command_line.config:show",
                "parakeet.config.new=parakeet.command_line.config:new",
                "parakeet.config.edit=parakeet.command_line.config:edit",
                "parakeet.sample.new=parakeet.command_line.sample:new",
                "parakeet.sample.add_molecules=parakeet.command_line.sample:add_molecules",
                "parakeet.sample.mill=parakeet.command_line.sample:mill",
                "parakeet.sample.sputter=parakeet.command_line.sample:sputter",
                "parakeet.sample.show=parakeet.command_line.sample:show",
                "parakeet.simulate.potential=parakeet.command_line.simulate:potential",
                "parakeet.simulate.exit_wave=parakeet.command_line.simulate:exit_wave",
                "parakeet.simulate.optics=parakeet.command_line.simulate:optics",
                "parakeet.simulate.image=parakeet.command_line.simulate:image",
                "parakeet.simulate.simple=parakeet.command_line.simulate:simple",
                "parakeet.simulate.cbed=parakeet.command_line.simulate:cbed",
                "parakeet.simulate.ctf=parakeet.command_line.simulate:ctf",
                "parakeet.metadata.export=parakeet.command_line.metadata:export",
                "parakeet.analyse.reconstruct=parakeet.command_line.analyse:reconstruct",
                "parakeet.analyse.average_particles=parakeet.command_line.analyse:average_particles",
                "parakeet.analyse.average_extracted_particles=parakeet.command_line.analyse:average_extracted_particles",
                "parakeet.analyse.average_all_particles=parakeet.command_line.analyse:average_all_particles",
                "parakeet.analyse.extract=parakeet.command_line.analyse:extract",
                "parakeet.analyse.refine=parakeet.command_line.analyse:refine",
                "parakeet.analyse.correct=parakeet.command_line.analyse:correct",
                "dev.parakeet.calibrate_ice_model=parakeet.util.calibrate_ice_model:main",
            ]
        },
        extras_require={
            "build_sphinx": ["sphinx", "sphinx_rtd_theme", "sphinx-argparse"],
            "test": tests_require,
        },
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
