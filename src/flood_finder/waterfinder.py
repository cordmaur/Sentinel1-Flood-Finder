"""
Implements the FloodSeeker class
This module requires scikit-learn == 1.2.2
"""
import logging
from typing import Optional, Union
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor

from tqdm.notebook import tqdm

from shapely import Polygon
import xarray as xr
import rioxarray as xrio  # pylint: disable=unused-import
from .imagery import ImageFinder
from .logging import create_logger
from .utils import open_tif_as_dset


class WaterFinder:
    """Docstring"""

    def __init__(
        self,
        output_path: Union[str, Path],
        aoi: Polygon,
        subscription_key: str,
        time_range: Optional[str] = None,
        lee_size: Optional[int] = 7,
        print_log: bool = False,
        log_level: int = logging.DEBUG,
        max_nodata: float = 0.05,
    ):
        """
        WaterFinder is a class that creates water masks from S1 imagery pulled from
        the Microsoft Planetary Computer

        Args:
            output_path (Union[str, Path]): Folder to write the masks
            aoi (Polygon): Area of Interest as a Shapely Polygon
            subscription_key(str): Planetary computer subscription key
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

        self.image_finder = ImageFinder(subscription_key=subscription_key)

        self.logger.info("Retrieving water recurrence")
        self.recurrence = self.image_finder.get_water_baseline(
            aoi=aoi, asset="recurrence"
        ).compute()

        self.logger.info("Retrieving dates list for the AOI")
        self.s1imagery = self.image_finder.get_s1_images(
            aoi=aoi,
            time_range=time_range,
            lee_size=lee_size,
            shape=self.recurrence.shape,
        )

        if (self.output_path / "waters.tif").exists():
            self.waters = open_tif_as_dset(self.output_path / "waters.tif")
            self.waters = self.adjust_coords(self.waters)

        else:
            self.waters = xr.Dataset()

    def adjust_coords(self, da: xr.DataArray) -> xr.DataArray:
        """
        Make sure the array has the same coords as self.permanent_water
        Additionally, get rid of unimportant coords
        Args:
            da (xr.DataArray): DataArray to be ajusted

        Returns:
            xr.DataArray: DataArray with the same coordinates as permanent_water
        """
        da = da.assign_coords({"x": self.recurrence.x, "y": self.recurrence.y})
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

    def find_water(
        self, model_path: Optional[Union[str, Path]] = None, resume: bool = True
    ):
        """
        Find water for the desired period. If resume==True, it will not override
        any previous detection.
        """
        classifier = None

        for i, date in enumerate(tqdm(self.s1imagery.dates)):
            if not resume or date not in self.waters:
                if classifier is None:
                    if model_path is None:
                        raise ValueError("A model is required to classify pixels")
                    classifier = joblib.load(model_path)

                try:
                    self.waters[date] = self.seek(date, classifier)
                except Exception:  # pylint: disable=W0718
                    self.logger.warning(  # pylintL disable=W1203
                        f"Date {date} has nodata greater than {100*self.max_nodata:.1f}% "
                    )

                if i % 10 == 9:
                    self.waters = self.waters.rio.set_crs("epsg:4326").astype("int")
                    self.waters.rio.to_raster(
                        self.output_path / "waters.tif", compress="DEFLATE"
                    )

        if classifier is not None:
            self.waters = self.waters.rio.set_crs("epsg:4326").astype("int")
            self.waters.rio.to_raster(
                self.output_path / "waters.tif", compress="DEFLATE"
            )

    def seek(self, date: str, classifier) -> xr.DataArray:
        """_summary_

        Args:
            date (str): _description_

        Returns:
            xr.DataArray: XArray containing the flood pixels
        """
        self.logger.info("Seeking for flood in date: %s", date)

        # get the corresponding image
        s1img = self.s1imagery[date]

        # check if there is "no-data" values in the image
        nodata_perc = s1img.isnull().sum() / s1img.size
        if nodata_perc > self.max_nodata:
            raise ValueError(f"Image has {nodata_perc*100:.1f}% no data values")

        water = WaterFinder.predict_water(classifier, s1img, thresh=0.5)
        water = self.adjust_coords(water).astype("int").rio.write_nodata(0)

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


# def calc_floods(clf, ref_img, imgs, bounds, crs, lee_size):

#     baseline = calc_baseline(ref_img, clf, bounds, crs, lee_size)

#     floods = []
#     preds = []
#     for i, img in enumerate(imgs):

#         actual = predict_water(clf, img, thresh=0.5, lee_size=lee_size)
#         flooded = (actual.astype('int') - baseline.astype('int')) == 1

#         # clean the prediction
#         kernel = skimage.morphology.square(5)
#         flooded.data = skimage.morphology.opening(flooded.data, footprint=kernel)

#         floods.append(flooded)
#         preds.append(actual)

#     return floods, preds, baseline
