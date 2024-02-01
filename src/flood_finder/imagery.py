"""Docstring"""
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Timedelta
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


import planetary_computer as pc
import pystac
import pystac_client
import stackstac
from shapely.geometry import Polygon
import xarray as xr
import rioxarray as xrio  # pylint: disable=unused-import


import requests


class S1RTCItem:
    """Wrapper for a S1-RTC Stac Item"""

    def __init__(self, item: pystac.Item):
        self.item = item
        self._grd = None
        self._manifest = None

    @property
    def grd(self):
        """Return the corresponding grd stac item"""
        if self._grd is None:
            # get the GRD link
            grdlnk = self.item.links[
                list(map(lambda x: x.rel, self.item.links)).index("derived_from")
            ]

            # load the GRD Item from the link
            self._grd = pc.sign(pystac.Item.from_file(href=grdlnk.href))

        return self._grd

    @property
    def manifest(self):
        """Get the XML manifest as an Element Tree"""
        if self._manifest is None:
            # read the manifest
            response = requests.get(
                self.grd.assets["safe-manifest"].href
            )  # pylint: disable=W3101
            self._manifest = ET.fromstring(response.content.decode("utf-8"))

        return self._manifest

    @property
    def IPF_version(self):  # pylint: disable=C0103
        """
        Return the version of the Instrument Processing Facility
        Versions higher than 2.9 does not need any denoising process
        """
        # Define the namespace
        namespace = {
            "safe": "http://www.esa.int/safe/sentinel-1.0",
        }

        # Search for the software element
        version = self.manifest.find(
            ".//safe:software[@name='Sentinel-1 IPF']", namespaces=namespace
        )

        # Check if the element is found
        return version.get("version")

    def __repr__(self):
        return str(self.item)


class S1Imagery:
    """Docstring"""

    # specify the time delta to group the images as one instance
    TIME_DELTA = Timedelta("12h")

    def __init__(
        self,
        items: List[pystac.Item],
        aoi: Polygon,
        lee_size: Optional[int] = None,
        shape: Optional[Tuple[int, int]] = None,
        noise_thresh: float = 2e-3,
    ):
        """
        The S1Imagery class is responsible for fetching S1Imagery from Planetary Computer.
        It groups the imagery by nearby date.
        Normaly, S1Imagery is not created directly, but it is instanced from the ImageFinder
        Args:
            items (List[pystac.Item]): STAC Igems
            aoi (Polygon): Area of Interest as a Shapely Polygon
            lee_size (Optional[int]): Size of the Lee Filter in pixels. Of it is set to "None"
            the filter is not applied, otherwise it is applied automatically during the fetching
            noise_thresh (float): Values below this threshold will be masked as no-data
        """
        self.items = items
        self.aoi = aoi
        self.lee_size = lee_size
        self.shape = shape
        self.noise_tresh = noise_thresh
        self.df = S1Imagery.create_grouped_df(items, S1Imagery.TIME_DELTA)

    @property
    def dates(self):
        """Return the `dates` available for this AOI"""
        return self.df["date"].unique()

    def plot_date(self, date: str, ax: Optional[plt.Axes] = None, raw: bool = False):
        """Plot a specific date"""
        if date not in self.dates:
            raise ValueError(f"Date {date} not in the list")

        # choose between RAW or Processed
        if raw:
            s1img = self.get_raw_date(date)
        else:
            s1img = self[date]

        rgb = S1Imagery.false_color_xr(s1img.sel(band="vv"), s1img.sel(band="vh"))
        rgb = rgb.clip(0, 1).fillna(0)

        if ax is None:
            _, ax = plt.subplots()

        rgb.plot.imshow(rgb="color_band", interpolation="antialiased", ax=ax)
        ax.set_title(date)

    @staticmethod
    def false_color(vv, vh):
        """
        Receives a 2D array and returns a 3D RGB array
        https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/sar_false_color_visualization/
        """

        c1 = 10e-4
        c2 = 0.01
        c3 = 0.02
        c4 = 0.03
        c5 = 0.045
        c6 = 0.05
        c7 = 0.9
        c8 = 0.25

        red = c4 + np.log(c1 - np.log(c6 / (c3 + 2 * vv)))
        green = c6 + np.exp(c8 * (np.log(c2 + 2 * vv) + np.log(c3 + 5 * vh)))
        blue = 1 - np.log(c6 / (c5 - c7 * vv))

        # rgb = np.stack([red, green, blue], axis=-1)
        return red, green, blue

    @staticmethod
    def false_color_xr(vv, vh):
        """Create a false color for S1 Images given vv and vh polarizations"""
        red, green, blue = S1Imagery.false_color(vv, vh)

        rgb_xr = xr.Dataset({"red": red, "green": green, "blue": blue})
        return rgb_xr.to_array(dim="color_band").fillna(0).clip(0, 1)

    def __len__(self):
        return len(self.dates)

    def get_raw_date(self, date: str):
        """_summary_

        Args:
            date (str): _description_

        Returns:
            _type_: _description_
        """
        items = self.df.query(f"date == '{date}'")["item"].to_list()

        cube = stackstac.stack(
            items=items, bounds_latlon=self.aoi.bounds, epsg=4326, chunksize=4096
        ).astype("float32")

        # remove fringe noise - refer to:
        # https://github.com/microsoft/PlanetaryComputer/issues/307
        cube = cube.where(cube.sel(band="vv") > self.noise_tresh)

        cube = cube.median(dim="time")
        return cube

    def __getitem__(self, idx: str):
        # items = self.df.query(f"date == '{idx}'")["item"].to_list()

        # cube = stackstac.stack(
        #     items=items, bounds_latlon=self.aoi.bounds, epsg=4326, chunksize=4096
        # ).astype("float32")

        # cube = cube.median(dim="time")

        cube = self.get_raw_date(idx)

        if self.lee_size is not None:
            # Calculate the reference
            cube = cube.clip(0, 10)
            cube = cube.fillna(0)
            cube.data = S1Imagery.lee_filter(cube, self.lee_size)
            cube = cube.where(cube > 0)

        cube = cube.rio.write_crs("epsg:4326")

        if self.shape:
            cube = cube.rio.reproject(dst_crs="epsg:4326", shape=self.shape)

        return cube

    def __iter__(self):
        for date in self.dates:
            yield date  # , self[date]

    @staticmethod
    def create_grouped_df(items: List[pystac.Item], time_delta: Timedelta):
        """Create a dataframe with the items and a group column, specified as the time smaller
        than the given time_delta.

        Args:
            items (List): list of STAC Items
            time_delta (Timedelta): period of time to group the images

        Returns:
            pd.DataFrame: Dataframe with the items and a `group` column
        """
        df = pd.DataFrame(
            {item.id: {"item": item, "datetime": item.datetime} for item in items}
        ).T
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["dt"] = df["datetime"] - df.shift(-1)["datetime"]
        df["delta"] = df["dt"] > time_delta
        df["group"] = (df["delta"]).cumsum().shift(1)
        df.iloc[0, -1] = 0
        df["group"] = df["group"].astype("int")

        # create the mean datetime as key to the group
        groups = df[["datetime", "group"]].groupby("group").mean()
        df["date"] = df["group"]
        groups["date"] = groups["datetime"].dt.strftime("%Y-%m-%d")

        df["date"] = df["date"].replace(groups["date"].to_dict())

        return df

    # defining a function to apply lee filtering on S1 image
    @staticmethod
    def lee_filter(da, size):
        """
        Apply lee filter of specified window size.
        Adapted from https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python

        """
        # get the kernel, considering the number of bands

        if "band" in da.dims:
            da = da.transpose("band", "y", "x")
            kernel = (1, size, size)
        else:
            kernel = (size, size)

        # create a mask
        mask = np.isnan(da.values)

        img = np.nan_to_num(da.values, nan=np.nanmean(da.values))

        img_mean = uniform_filter(img, kernel)
        img_sqr_mean = uniform_filter(img**2, kernel)
        img_variance = img_sqr_mean - img_mean**2

        overall_variance = variance(img)

        img_weights = img_variance / (img_variance + overall_variance)
        img_output = img_mean + img_weights * (img - img_mean)

        # write the mask
        img_output[mask] = np.nan

        return img_output

    def __repr__(self):
        """String representation of the object"""
        s = f"S1Imagery Class with {len(self)} dates\n"
        return s


class ImageFinder:
    """This class is responsible for querying and fetching imagery data from the planetary
    computer"""

    def __init__(self, subscription_key):
        pc.set_subscription_key(subscription_key)

        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

    def get_water_baseline(self, aoi: Polygon, asset: str) -> xr.DataArray:
        """Return the Global Surface Water product for the desired aoi"""

        search = self.catalog.search(collections=["jrc-gsw"], intersects=aoi)
        items = search.item_collection()
        cube = stackstac.stack(items=items, assets=[asset], bounds_latlon=aoi.bounds)
        cube = cube.where(cube > 0)
        cube = cube.median(dim="time").squeeze()

        return cube

    def get_s1_images(
        self,
        aoi: Polygon,
        time_range: Optional[str] = None,
        lee_size: Optional[int] = 7,
        shape: Optional[Tuple[int, int]] = None,
    ) -> S1Imagery:
        """
        Get a S1Imagery instance given an area and a time range STAC Item
        Args:
            aoi (Polygon): Area of interest. It should be a shapely Polygon.
            time_range (str): The period of time to search for images (e.g., 2016-01-01/2020-02-20)
        """

        search = self.catalog.search(
            collections=["sentinel-1-rtc"], intersects=aoi, datetime=time_range
        )

        items = list(search.item_collection())

        return S1Imagery(items, aoi, lee_size, shape)
