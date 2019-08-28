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
        install_requires=["numpy", "python-multem", "gemmi"],
        setup_requires=["pytest-runner"],
        tests_require=tests_require,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "elfantasma=elfantasma.command_line:main",
                "elfantasma-read-pdb=elfantasma.command_line:read_pdb",
            ]
        },
        extras_require={
            "build_sphinx": ["sphinx", "sphinx_rtd_theme"],
            "test": tests_require,
        },
    )


if __name__ == "__main__":
    main()
