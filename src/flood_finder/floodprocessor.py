"""FloodProcessor class"""
from typing import Union, Optional, List
from pathlib import Path
import logging
from dateutil.parser import parse

from tqdm.notebook import tqdm

from shapely import Geometry, box
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as xrio
import numpy as np
import skimage
import contextily as cx
from xyzservices import TileProvider

# from .utils import adjust_coords
from .imagery import ImageFinder
from .logging import create_logger
from .waterfinder import WaterFinder


class FloodProcessor:
    """_summary_"""

    def __init__(
        self,
        aoi_df: gpd.GeoDataFrame,
        output_dir: Union[str, Path],
        subscription_key: str,
        # dem: Union[str, Path],
        # hand: Union[str, Path],
        time_range: Optional[str] = None,
        lee_size: Optional[int] = 7,
        recurrence_threshold: int = 10,
        print_log: bool = False,
        log_level: int = logging.DEBUG,
    ):
        """_summary_

        Args:
            aoi_df (gpd.GeoDataFrame): _description_
            output_dir (Union[str, Path]): _description_
            subscription_key (str): _description_
            time_range (Optional[str], optional): _description_. Defaults to None.
            lee_size (Optional[int], optional): _description_. Defaults to 7.
            recurrence_threshold (int, optional): _description_. Defaults to 10.
            print_log (bool, optional): _description_. Defaults to False.
            log_level (int, optional): _description_. Defaults to logging.DEBUG.
        """
        # self.image_finder = ImageFinder(subscription_key=subscription_key)
        self.output_dir = Path(output_dir)
        self.recurrence_threshold = recurrence_threshold

        # make sure output_dir exists
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.name = self.output_dir.stem

        # first create a root logger for the AOI
        self.parent_logger = create_logger(
            name=self.name,
            level=log_level,
            folder=self.output_dir,
            fname=None,
            print_log=print_log,
        )

        # create a specific logger for the FloodFinder
        self.logger = logging.getLogger(self.name + "." + type(self).__name__)
        self.logger.info("Creating processor for place: %s", self.name)

        # first, let's save the basic variables in the output dir
        self.logger.info("Saving file %s", self.output_dir / "gdf.geojson")
        self.vars = {}
        aoi_df.to_file(self.output_dir / "gdf.geojson")
        self["aoi_df"] = aoi_df

        # create a container for the variables

        self.finder = WaterFinder(
            output_path=self.output_dir,
            aoi=self.aoi,
            subscription_key=subscription_key,
            time_range=time_range,
            lee_size=lee_size,
            print_log=print_log,
            log_level=log_level,
        )

        self.logger.info("Get water_recurrence from WaterFinder")
        self["recurrence"] = self.finder.recurrence

        # # load HAND and DEM models
        # self.logger.info("Loading DEM from %s", dem)
        # self["dem"] = self.load_window(
        #     dem, self.bounds, shape=self.vars["recurrence"].shape
        # )
        # self["dem"] = adjust_coords(self["dem"], self["recurrence"])

        # self.logger.info("Loading HAND from %s", hand)
        # self.vars["hand"] = self.load_window(
        #     hand, self.bounds, shape=self.vars["recurrence"].shape
        # )
        # self["hand"] = adjust_coords(self["hand"], self["recurrence"])

        self.logger.info("Saving variables locally")
        self.save_vars(["recurrence"])
        # self.save_vars(["recurrence", "dem", "hand"])

        # self.vars["urban_area"] = ua
        # self.vars["aoi"] = ua.poly
        # self.vars["dem"] = self.load_window(
        #     dem, self.bounds, shape=self.vars["waters"].shape[1:3]
        # )
        # self.vars["hand"] = self.load_window(
        #     hand, self.bounds, shape=self.vars["waters"].shape[1:3]
        # )
        # self.vars["recurrence"] = self.image_finder.get_water_baseline(
        #     ua.poly, asset="recurrence"
        # )

        # self.vars["recurrent_water"] = self.vars["recurrence"].where(
        #     self.vars["recurrence"] > 10
        # )

        # water_series = self["waters"].sum(dim=["x", "y"]).to_series()
        # water_series.index = [parse(date) for date in self["waters"].long_name]

        # self.vars["water_series"] = water_series

        # self.vars["max_water"] = self["waters"][water_series.argmax()]
        # flood_arr = (
        #     self["max_water"].where(self["hand"].data < 10).data
        #     - (self["recurrent_water"] > 0).data
        # ) == 1

        # # clean the prediction
        # kernel = skimage.morphology.square(5)
        # flood_arr = skimage.morphology.opening(flood_arr.compute(), footprint=kernel)
        # flooded = self["recurrent_water"].copy()
        # flooded.data = flood_arr
        # self.vars["max_flood"] = flooded

        # try:
        #     (
        #         self.vars["flooded_regions"],
        #         self.vars["labels"],
        #         self.vars["dem_steps"],
        #     ) = self.extrapolate_flood()
        #     floods = np.stack(self["flooded_regions"])
        #     self.vars["extrapolated_flood"] = self["dem"].copy()
        #     self.vars["extrapolated_flood"].data = floods.any(axis=0).astype("int")
        # except Exception as e:  # pylint: disable=broad-except
        #     print(e)

    @property
    def bounds(self):
        """Return the bounds for the area of interest by applying a buffer around it"""
        return self["aoi_df"].total_bounds
        # return calc_bounds(self.vars["aoi"], percent_buffer=0.0)

    @property
    def aoi(self) -> Geometry:
        """Return the AOI as a shapely Geometry"""
        return box(*self.bounds)

    def save_vars(self, vars_lst: List[str]):
        """Save GeoTiff for each var in the list"""

        for var in vars_lst:
            if var in self.vars:
                tif = self[var]
                name = self.name + "_" + var + ".tif"
                tif.rio.to_raster(self.output_dir / name, compress="DEFLATE")

    def process_floods(self, recurrence_threshold: int = 10, use_hand: bool = False):
        """Create the floods for each date and save the table to csv
        Args:
            recurrence_threshold (int, optional): Minimum recurrence value to consider the pixel
            as a permanent water. Defaults to 10.
            use_hand (bool): If True, use the HAND model to avoid false positives or flood
            not related to river overflow. Defaults to True.
        """
        self.logger.info("Calculating flood area for each date")

        floods = xr.Dataset()

        for date in tqdm(self["water_series"].index.astype("str")):
            flood = self.find_flood(
                date, recurrence_threshold=recurrence_threshold, use_hand=use_hand
            )
            flood_area = int(flood.sum()) * 0.0009

            self["data_table"].loc[parse(date), "Flood area"] = flood_area
            floods[date] = flood

        self["data_table"].to_csv(self.output_dir / "table.csv")
        self.logger.info("table.csv exported with water/flood series")

        # fill NaN with 0s
        floods = floods.fillna(0)
        floods = floods.rio.set_crs("epsg:4326").astype("int")
        floods.rio.to_raster(self.output_dir / "floods.tif", compress="DEFLATE")

        # save the max flood
        date_max = self["data_table"]["Flood area"].idxmax()
        self["max_flood"] = floods[date_max.strftime("%Y-%m-%d")]
        self["max_flood"] = self["max_flood"].where(self["max_flood"] > 0)

        self["floods"] = floods

        # self.logger.info("calculating the extrapolated flood")

        # ## Extrapolate Flood through DEM/HAND
        # try:
        #     (
        #         self.vars["flooded_regions"],
        #         self.vars["labels"],
        #         self.vars["dem_steps"],
        #     ) = self.extrapolate_flood()
        #     floods = np.stack(self["flooded_regions"])
        #     self.vars["extrapolated_flood"] = self["recurrence"].copy()
        #     self.vars["extrapolated_flood"].data = floods.any(axis=0).astype("int")

        #     self.vars["extrapolated_flood"].rio.to_raster(
        #         self.output_dir / "extrapolated_flood.tif", compress="DEFLATE"
        #     )

        #     self["vulnerable"] = (
        #         self["extrapolated_flood"] - (self["recurrence"] > recurrence_threshold)
        #     ) > 0

        #     # create the urban_vulnerable variable
        #     vul = self["vulnerable"].astype("int").rio.write_nodata(0)
        #     vul = vul.rio.set_crs("epsg:4326")

        #     self["urban_vul"] = vul.rio.clip(self["aoi_df"].geometry)

        #     self.save_vars(["extrapolated_flood"])

        # except Exception as e:  # pylint: disable=broad-except
        #     print(e)

    def find_flood(
        self,
        date: str,
        recurrence_threshold: int = 10,
        size_threshold: int = 25,
        use_hand: bool = False,
    ):
        """Find flood for a specific date

        Args:
            date (str): _description_
            recurrence_threshold (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        water = self["waters"][date]

        if use_hand:
            if "hand" not in self.vars:
                raise ValueError(
                    "HAND model not available. Try to process floods without HAND."
                )
            water_data = water.where(self["hand"].data < 10).data
        else:
            water_data = water.data

        flood_arr = (water_data - (self["recurrence"] > recurrence_threshold).data) == 1

        # clean the prediction
        # kernel = skimage.morphology.square(2)
        # flood_arr = skimage.morphology.opening(flood_arr, footprint=kernel)
        # flood_arr = skimage.morphology.remove_small_objects(
        #     flood_arr, min_size=size_threshold
        # )
        flooded = self["recurrence"].copy()
        flooded.data = flood_arr

        return flooded.where(flooded > 0)

    def find_water(
        self,
        model_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
    ):
        """
        The find_water method creates a `water.tif` file with water detected for
        every date in the time_range. It uses a RandomForests regressor.

        Args:
            model (Union[str, Path]): path of the RandomForests regressor model
            time_range (str): time range to search for water
            lee_size (int, optional): Size of lee filter to be applied. Defaults to 7.
            resume (bool, optional): If True, it will try to recover from previous detection.
            If False, it will override the `water.tif`. Defaults to True.
        """

        self.logger.info("Creating a water extents series")

        self.finder.find_water(model_path=model_path, resume=resume)
        self["waters"] = self.finder.waters

        water_series = (
            self["waters"].to_array(dim="time").sum(dim=["x", "y"]).to_series()
        )
        water_series.index = [parse(date) for date in water_series.index]
        self["water_series"] = water_series * 0.0009  # conversion to km2
        self["data_table"] = self["water_series"].rename("Water Extents").to_frame()

    def load_window(self, geotiff_path: Union[str, Path], bounds: tuple, shape: tuple):
        """Load a window from a geotiff file"""

        xmin, ymin, xmax, ymax = bounds

        tif = xrio.open_rasterio(geotiff_path).squeeze()  # type: ignore
        window = tif.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)).compute()

        result = window.where(window != window.attrs["_FillValue"])
        result = result.rio.reproject(dst_crs="epsg:4326", shape=shape)
        return result

    def plot_var(self, var: str, ax: plt.Axes, **kwargs):
        """Plot a single var within the given axes"""
        self[var].plot(ax=ax, **kwargs)

    def plot_recurrent_water(self, ax: plt.Axes, dem: str = "dem"):
        """Plot the area with the recurrent water"""
        self.plot_var(dem, ax=ax, cmap="gist_earth", vmin=0, add_colorbar=False)
        self.plot_var("aoi_df", facecolor="none", edgecolor="white", ax=ax)
        self.plot_var("recurrence", cmap="cool_r", add_colorbar=False, ax=ax)

    def plot_vars(self, ax: plt.Axes, dem: str = "dem") -> None:
        """Plot all variables in the same Axes"""
        self.plot_var(dem, ax=ax, cmap="gist_earth", vmin=0, add_colorbar=False)
        # self.plot_var("ref", vmax=1, cmap="Blues", ax=ax, add_colorbar=False)
        # self.plot_var("max_flood", vmax=1, cmap="brg", ax=ax, add_colorbar=False)
        self.plot_var("urban_area", facecolor="none", edgecolor="white", ax=ax)
        self.plot_var("recurrent_water", cmap="cool_r", add_colorbar=False, ax=ax)

    def plot_flood(
        self,
        date: str,
        ax: Optional[plt.Axes] = None,
        recurrence_threshold: int = 10,
        background: Optional[Union[str, TileProvider]] = "dem",
    ):
        """Plot the flood for a specific date

        Args:
            ax (plt.Axes): Matplotlib Axes
            date (str): Date in string
            recurrence_threshold (int, optional):Defaults to 10.
            background (str, TileProvider, optional): As background we can plot the DEM or HAND (if available).
            Or we can pass a contextily.provider. Defaults to "dem".
        """

        # get the Axes
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        recurrent_water = self["recurrence"].where(
            self["recurrence"] > recurrence_threshold
        )

        recurrent_water.plot.imshow(
            ax=ax,
            cmap="Blues",
            zorder=1,
            add_colorbar=False,
        )

        flood = self.find_flood(date, recurrence_threshold=recurrence_threshold)
        flood.plot.imshow(ax=ax, cmap="Reds", zorder=2, add_colorbar=False)

        # if the background is DEM or HAND, plot them with cmap gist_earth
        if background is not None:
            if isinstance(background, TileProvider):
                cx.add_basemap(ax=ax, crs="epsg:4326", source=background)

            elif background.lower() in ["dem", "hand"]:
                self.plot_var(
                    background,
                    ax=ax,
                    vmin=0,
                    cmap="gist_earth",
                    add_colorbar=False,
                    zorder=0,
                )

            else:
                raise ValueError(f"Background {background} is not supported")

        self.plot_var("aoi_df", ax=ax, facecolor="none", edgecolor="white")

        ax.set_title(f"Flood for {date}")

        return ax

    def plot_extrapolated_flood(self, ax: plt.Axes):
        """_summary_

        Args:
            ax (plt.Axes): _description_
        """
        self["extrapolated_flood"].where(self["extrapolated_flood"] > 0).plot(
            ax=ax, add_colorbar=False, vmin=0
        )

        self["urban_vul"].where(self["urban_vul"] > 0).plot(
            ax=ax, add_colorbar=0, cmap="Reds"
        )

        self.plot_var("aoi_df", ax=ax, facecolor="none", edgecolor="black")

    @staticmethod
    def flood_areas_hand(flood_mask: np.ndarray, dem: np.ndarray, hand: np.ndarray):
        """
        Extrapolate the obtained flood mask according to the DEM and HAND of the area.
        This function operate on a pixel basis (raster), so all paramaters must be given in
        Numpy arrays with the same shape.
        """
        # isolate all the identified flood areas
        labels = skimage.measure.label(flood_mask)
        # print(f'Number of areas to flood: {labels.max()}')

        # create lists to store the floods for each label and the dem region for each label
        floods = []
        dem_steps = []

        # let's loop through each label (i.e., flood region)
        for label in range(1, labels.max() + 1):  # type: ignore
            # print(f'Processing label {label}')

            # get the flood for the corresponding label (area)
            # and set all other pixels to 0
            flood_step = labels.copy()  # type: ignore
            flood_step[labels != label] = 0

            # get the highest pixel within the area, but try to remove any outlier
            height = np.percentile(dem[labels == label], 95)
            # height = dem[labels==label].max()
            # print(f'Height={height}')

            # create a the DEM-fences with the calculated height
            # this guarantees the flood fill will not go uphill and will not cross boundaries
            dem_step = dem.copy()
            dem_step[dem_step <= height] = 0
            dem_step[dem_step > height] = 1

            # the problem with the last assumption is that the river goes down so the farthest from the fill point
            # a bigger area will be flooded. In this case, we add a second assumption considering the HAND value
            hand_height = np.percentile(hand[labels == label], 95)
            if not np.isnan(hand_height):
                # print(f'Hand height = {hand_height}')
                # dem_step[hand <= hand_height] = 0
                dem_step[hand > hand_height] = 1

                dem_steps.append(dem_step)

                # to flood-fill, we need a starting point, we can get the lowest point with the label
                xs, ys = np.where(labels == label)
                pos = dem[xs, ys].argmin()
                start = (xs[pos], ys[pos])

                # flood fill and get the extended flood for this label
                flood = skimage.morphology.flood_fill(
                    dem_step, seed_point=start, new_value=-1
                )
                flood = np.where(flood == -1, 1, 0)

            else:
                print("No hand available")
                flood = np.where(labels == label, 1, 0)
                dem_steps.append(flood)

            floods.append(flood)

        return floods, labels, dem_steps  # type: ignore

    def extrapolate_flood(self, size_threshold: int = 25):
        """Extrapolate the floods according to the DEM and HAND"""

        # first, we clean the area by removing very small regions
        # our pixel is 30x30m. A threshold of 25 will ensure a minimum area of
        # 2.5ha for each flooded region
        # max_flood = np.nan_to_num(self["max_flood"].data, nan=0)
        # flood_mask = skimage.morphology.area_opening(
        #     max_flood, area_threshold=threshold
        # )

        date_max = self["data_table"].index.astype("str")[
            self["data_table"]["Water Extents"].argmax()
        ]
        # date_max = '2017-06-11'
        max_flood = self.find_flood(
            date_max, recurrence_threshold=10, size_threshold=size_threshold
        )
        max_flood = np.nan_to_num(max_flood.data, nan=0).astype("bool")

        # já foi feita a limpeza durante o find_flood
        # flood_mask = skimage.morphology.area_opening(
        #     max_flood, area_threshold=threshold
        # )
        flood_mask = skimage.morphology.remove_small_objects(
            max_flood, min_size=size_threshold
        )

        self["max_flood"] = self["recurrence"].copy()
        self["max_flood"].data = max_flood.astype("int")
        self["max_flood"] = self["max_flood"].where(self["max_flood"] > 0)

        # check if there is at least 1 region to process
        if not flood_mask.any():
            raise ValueError(
                f"No flooded regions considering threshold={size_threshold}"
            )

        (
            flooded_regions,
            labels,
            dem_steps,
        ) = FloodProcessor.flood_areas_hand(
            flood_mask, self["dem"].data, self["hand"].data
        )

        return flooded_regions, labels, dem_steps

    def __getitem__(self, idx):
        """Get a variable from the processor"""
        return self.vars[idx]

    def __setitem__(self, idx, value):
        """Set a variable to the processor"""
        self.vars[idx] = value

    def __repr__(self):
        s = "Flood Processor for place: "
        s += str(self.name)
        s += "\nVariables: " + str(list(self.vars.keys()))
        s += "\n" + str(self.finder)
        return s
