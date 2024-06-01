"""
Implements the FloodSeeker class
This module requires scikit-learn == 1.2.2
"""

from collections.abc import Iterable
from io import BytesIO
from time import sleep
import logging
from typing import Optional, Union, Tuple
from pathlib import Path

import joblib
from owslib.wms import WebMapService
from PIL import Image

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from tqdm.notebook import tqdm

from shapely import Polygon
import xarray as xr
import rioxarray as xrio  # pylint: disable=unused-import
from .imagery import ImageFinder
from .logging import create_logger


class WaterFinder:
    """Docstring"""

    GFM_URL = "https://geoserver.gfm.eodc.eu/geoserver/gfm/wms"
    GFM_LAYER = "observed_flood_extent"

    def __init__(
        self,
        output_path: Union[str, Path],
        aoi: Polygon,
        group_items: bool = False,
        time_range: Optional[str] = None,
        lee_size: Optional[int] = 7,
        print_log: bool = False,
        log_level: int = logging.DEBUG,
        max_nodata: float = 0.05,
        shape: Optional[Tuple[int, int]] = None,
    ):
        """
        WaterFinder is a class that creates water masks from S1 imagery pulled from
        the Microsoft Planetary Computer

        Args:
            output_path (Union[str, Path]): Folder to write the masks to
            aoi (Polygon): Area of Interest as a Shapely Polygon
            group_items (bool): if True, will create a cube with all items and calculate the median
            time_range (Optional[str], optional): Time frame as a string
                For example, '2016/2020' or '2020-01-01/2020-02-01', etc..
                If None, the whole period is considered: Defaults to None.
            lee_size (Optional[int], optional): Number of pixels for the lee filter. Defaults to 7.
            print_log (bool, optional): If log is printed to standard output. Defaults to False.
            log_level (int, optional): Logging level. Defaults to logging.DEBUG.
            max_nodata (float, optional): Maximum percentage of no data value in the area to
                create the mask. Defaults to 0.05 (5%).
        """

        self.max_nodata = max_nodata

        # first create a root logger for the AOI
        self.output_path = Path(output_path)

        self.parent_logger = create_logger(
            name=self.output_path.stem,
            level=log_level,
            folder=output_path,
            fname=None,
            print_log=print_log,
        )

        # create a specific logger for the FloodFinder
        self.logger = logging.getLogger(
            self.output_path.stem + "." + type(self).__name__
        )

        self.logger.info("Starting WaterFinder instance for %s", self.output_path.name)

        self.image_finder = ImageFinder()

        self.logger.info("Retrieving dates list for the AOI")
        self.s1imagery = self.image_finder.get_s1_images(
            aoi=aoi,
            time_range=time_range,
            lee_size=lee_size,
            shape=shape,
            group_items=group_items,
        )

        self.waters = xr.Dataset().rio.set_crs("epsg:4326")

    def adjust_coords(self, da: xr.DataArray) -> xr.DataArray:
        """
        Make sure the array has the same coords as imagery template
        Additionally, get rid of unimportant coords
        Args:
            da (xr.DataArray): DataArray to be ajusted

        Returns:
            xr.DataArray: DataArray with the same coordinates as permanent_water
        """
        template = self.s1imagery.template
        da = da.assign_coords({"x": template.x, "y": template.y})
        drop_vars = set(da.coords.keys())
        drop_vars = drop_vars - {"x", "y", "epsg"}
        da = da.drop_vars(drop_vars)
        return da

    @property
    def dates_range(self):
        """Return the period considered to detect water"""
        s = self.s1imagery.dates[-1]
        s += " / " + self.s1imagery.dates[0]
        return s

    def find_water_in_dates(
        self,
        dates: list,
        model_path: Optional[Union[str, Path]] = None,
        use_gfm: bool = False,
    ):
        """
        Main loop to find water in the specified dates
        Result is stored in the `self.waters`
        """
        if not isinstance(dates, Iterable):
            raise ValueError("Dates argument should be an Iterable obj")

        if len(dates) > 0:
            if use_gfm:
                wms_t = WebMapService(WaterFinder.GFM_URL)
                seek_fn = self.seek_gfm
                args = {"wms_t": wms_t}

            else:
                if model_path is None:
                    raise ValueError("You must pass a RF model path")

                classifier = joblib.load(model_path)
                seek_fn = self.seek
                args = {"classifier": classifier}

            # loop through the dates
            for i, date in enumerate(tqdm(dates, desc=self.output_path.stem)):
                try:
                    # call the corresponding function
                    self.waters[date] = seek_fn(date, **args)

                except Exception as e:  # pylint: disable=W0718
                    self.logger.warning(e)

                if (i % 10 == 9) or (i == len(dates) - 1):
                    self.logger.info(  # pylint: disable=W1203
                        f"Saving waters.nc. i={i}"
                    )

                    encoding = {
                        var: {"dtype": "uint8", "zlib": True}
                        for var in self.waters.data_vars.keys()
                    }
                    self.waters.to_netcdf(
                        self.output_path / "waters.nc", encoding=encoding
                    )

        else:
            print("No dates to classify")

    def find_water_in_date(
        self,
        date: str,
        model_path: Optional[Union[str, Path]] = None,
        use_gfm: bool = True,
    ) -> xr.DataArray:
        """
        Find water in a single date

        Args:
            date (str): Date as string
            model_path (Optional[Union[str, Path]]): Path to the RF model
            use_gfm (bool, optional): If True, will use the GFM WMS. Defaults to True.

        Returns:
            xr.DataArray: Water mask for the given date
        """
        # check if the date is available
        if date not in self.s1imagery.dates:
            raise ValueError(f"Date {date} is not available")

        # call the corresponding function
        if use_gfm:
            wms_t = WebMapService(WaterFinder.GFM_URL)
            seek_fn = self.seek_gfm
            args = {"wms_t": wms_t}

        else:
            if model_path is None:
                raise ValueError("You must pass a RF model path to classify water")

            classifier = joblib.load(model_path)
            seek_fn = self.seek
            args = {"classifier": classifier}

        return seek_fn(date, **args)

    def find_water(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_gfm: bool = True,
        resume: bool = True,
    ):
        """
        Find water for the desired period. If resume==True, it will not override
        any previous detection.
        """

        # load previous waters.nc if it exists and if resume is True
        if resume and (self.output_path / "waters.nc").exists():
            self.logger.info("Resuming from previous waters.nc file")
            self.waters = xr.open_dataset(
                self.output_path / "waters.nc", mask_and_scale=False
            ).compute()
            self.waters = self.waters.rio.write_crs("epsg:4326")

            # Close the connection
            self.waters.close()

            # check for the last ingested date
            dates = sorted(list(self.waters.data_vars.keys()))
            if len(dates) > 0:
                last_date = dates[-1]
                self.logger.info("Resuming from date %s", last_date)

            else:
                last_date = ""

            # define the dates to loop through
            loop_dates = self.s1imagery.dates[self.s1imagery.dates > last_date]

        else:
            self.waters = xr.Dataset().rio.set_crs("epsg:4326")
            loop_dates = self.s1imagery.dates

        if len(loop_dates) > 0:
            return self.find_water_in_dates(
                model_path=model_path, dates=loop_dates, use_gfm=use_gfm
            )

        else:
            print("No date left to process. Check self.waters")

    @staticmethod
    def get_wms_img(
        time_range: str, layer: str, bbox: tuple, size: tuple, wms_t: WebMapService
    ) -> Image:
        """_summary_

        Args:
            time_range (str): _description_
            layer (str): _description_
            bbox (tuple): _description_
            size (tuple): _description_
            wms_t (WebMapService): _description_

        Returns:
            Image: _description_
        """

        retries = 5

        while retries:
            try:
                wms_request = wms_t.getmap(
                    layers=[layer],
                    styles="",
                    srs="EPSG:4326",
                    bbox=bbox,
                    size=size,
                    time=time_range,
                    format="image/png",
                )
                obj = BytesIO(wms_request.read())
                img = Image.open(obj)
                return img

            except Exception as e:  # pylint: disable=W0718
                sleep(1)
                retries = retries - 1
                print(f"Error fetching {time_range}. {retries} retries left\n{e}")

                # restart the connection
                wms_t = WebMapService(wms_t.url)

    def seek_gfm(self, date: str, wms_t: WebMapService) -> xr.DataArray:
        """_summary_

        Args:
            date (str): _description_

        Returns:
            xr.DataArray: XArray containing the flood pixels
        """

        self.logger.info("Seeking for GFM flood in date: %s", date)

        shape = self.s1imagery.shape
        img = WaterFinder.get_wms_img(
            time_range=date,
            layer=WaterFinder.GFM_LAYER,
            size=(shape[1], shape[0]),
            bbox=self.s1imagery.aoi.bounds,
            wms_t=wms_t,
        )
        sleep(0.5)

        if img is None:
            raise ValueError(f"Image for {date} not retrieved.")

        array = np.array(img).astype("uint8")
        if len(array.shape) == 3:
            raise ValueError(f"No data for date {date}.")

        water = self.s1imagery.template.copy()
        water.data = array

        water = self.adjust_coords(water).astype("uint8").rio.write_nodata(0)

        return water

    def seek(self, date: str, classifier) -> xr.DataArray:
        """_summary_

        Args:
            date (str): _description_

        Returns:
            xr.DataArray: XArray containing the flood pixels
        """
        self.logger.info("Seeking for water in date: %s", date)

        # get the corresponding image
        s1img = self.s1imagery[date]

        # check if there is "no-data" values in the image
        nodata_perc = s1img.isnull().sum() / s1img.size
        if nodata_perc > self.max_nodata:
            raise ValueError(f"Image has {nodata_perc*100:.1f}% no data values")

        water = WaterFinder.predict_water(classifier, s1img, thresh=0.45)
        water = self.adjust_coords(water).astype("uint8").rio.write_nodata(0)

        # flood_arr = (water.data - self.permanent_water.astype("int").data) == 1

        # # clean the prediction
        # kernel = skimage.morphology.square(5)
        # flood_arr = skimage.morphology.opening(flood_arr, footprint=kernel)
        # flooded = self.permanent_water.copy()
        # flooded.data = flood_arr

        # # compat the coords
        # flooded = self.adjust_coords(flooded).rio.write_nodata(0)
        # water = self.adjust_coords(water).rio.write_nodata(0)

        # return flooded, water
        return water

    @staticmethod
    def predict_water(
        clf: RandomForestRegressor,
        img: xr.DataArray,
        thresh: float = None,
    ):
        """_summary_

        Args:
            clf (RandomForestRegressor): _description_
            img (xr.DataArray): _description_
            thresh (float, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Calculate the reference
        tmp = img.clip(0, 10).copy()
        tmp = tmp.fillna(10)

        data = tmp.data.reshape(2, -1).transpose(1, 0)
        probs = clf.predict_proba(data).transpose(1, 0).reshape(img.shape)

        result = img.sel(band="vv").copy()
        if thresh is None:
            result.data = probs
        else:
            result.data = probs[1] > thresh

        return result

    def __repr__(self):
        s = f"WaterFinder for place {self.output_path.stem}\n"
        s += f"Available dates: {len(self.s1imagery)}\n"
        s += f"Water detected dates: {len(self.waters)}"
        return s
