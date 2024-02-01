"""Setup file"""
import sys
from setuptools import setup, find_packages

sys.path.append("./src")
import flood_finder  # pylint: disable=E0401, C0413

setup(
    name="S1FloodFinder",
    version=flood_finder.__version__,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    entry_points={},
)
