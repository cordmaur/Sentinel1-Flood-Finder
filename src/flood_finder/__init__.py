"""Flood Finder package"""

import warnings

from .imagery import ImageFinder, S1Imagery
from .waterfinder import WaterFinder


# Filter out warnings from specific module
warnings.filterwarnings("ignore", category=UserWarning, module="stackstac")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="xarray")
warnings.filterwarnings("ignore", module="img2pdf")

__version__ = "0.0.1"
