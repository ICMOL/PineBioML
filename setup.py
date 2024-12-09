"""Module setup."""

import runpy
from setuptools import setup, find_packages

PACKAGE_NAME = "PineBioML"
version_meta = runpy.run_path("./PineBioML/__init__.py")
VERSION = version_meta["__version__"]

with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


if __name__ == "__main__":
    setup(name=PACKAGE_NAME,
          version=VERSION,
          url="https://github.com/ICMOL/PineBioML",
          packages=find_packages(),
          install_requires=parse_requirements("requirements.txt"),
          python_requires=">=3.6.0",
          description="PineBioML is a easy use ML toolkit.",
          long_description=long_description,
          long_description_content_type="text/markdown",
          classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent",
          ])
    # to publish
    # requirement: setuptools twine wheel
    # remove early version in dist/*
    #
    # python setup.py sdist bdist_wheel
    # python -m twine check dist/*
    # python -m twine upload dist/*
