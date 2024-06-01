"""Urban Areas Module"""

from typing import Union, Tuple, Iterable, Dict
from pathlib import Path
import unidecode
import json
import geopandas as gpd
from shapely.geometry import box, Polygon
from .utils import calc_aspects_lims


class UrbanArea:
    """Docstring"""

    def __init__(self, aoi_df: gpd.GeoDataFrame, bounds: gpd.GeoDataFrame):
        self.df = aoi_df
        self.bounds = bounds

    @property
    def name(self):
        """Docstring"""
        return self.df.iloc[0]["NM_MUN"]

    @property
    def uf(self):
        """Docstring"""
        return self.df.iloc[0]["SIGLA"]

    @property
    def area(self):
        """Docstring"""
        return int(self.df["area_km2"].sum())

    # @property
    # def bounds(self):
    #     """Docstring"""
    #     return self.df.total_bounds

    @property
    def poly(self) -> Polygon:
        """Return a polygon with the bounds"""
        return box(*self.bounds)

    @property
    def json(self) -> Dict:
        """Return the urban area as a GEO-JSON object"""
        return json.loads(self.df.to_json())

    @property
    def cd_mun(self) -> int:
        """Return the Municipal Code"""
        return self.df["CD_MUN"].iloc[0]

    def get_folder(self, root: Union[str, Path]):
        """Create a folder to store the results of this urban area"""

        name = unidecode.unidecode(self.name).replace(" ", "_")
        path = Path(root) / name
        path.mkdir(exist_ok=True, parents=True)
        return path

    def plot(self, **kwargs):
        """Plot the GeoDataFrame"""
        return self.df.plot(**kwargs)

    @property
    def total_area(self) -> float:
        """Calculate the total area of the bounds in km2"""
        return self.bounds.to_crs("esri:54034").area[0] * 1e-6

    def __repr__(self):
        return f"Urban area for: {self.name}"


class UrbanAreas:
    """
    Wraps up functions to process the UrbanAreas shapefile from Brazil's IBGE
    Its main functionality is to allow quick Zoom-in into the biggest urban area of a municipality
    If using another shapefile format, refer directly to the `calc_aspects_lims` from `utils.py`
    """

    def __init__(self, urban_areas_shp: Union[str, Path], crs: str = "epsg:4326"):
        self.urban_areas_path = Path(urban_areas_shp)
        self.crs = crs
        self.urban_areas = gpd.read_file(urban_areas_shp).to_crs(crs)

        UrbanAreas.update_area(self.urban_areas)

    def __repr__(self) -> str:
        s = "Urban Areas class with the following file loaded:\n"
        s += str(self.urban_areas_path)
        return s

    def get_city(self, city: Union[int, str]) -> gpd.GeoDataFrame:
        """Return the dataframe corresponding to a specific city"""
        if isinstance(city, int):
            df = self.urban_areas.query(f"CD_MUN == '{str(city)}'")

        elif isinstance(city, str):
            df = self.urban_areas.query(f"NM_MUN == '{city}'")

        else:
            raise ValueError(
                "city value must be either `int` for the code or `string` for name"
            )

        if len(df) == 0:
            raise ValueError(f"City {city} not found in database")
        else:
            return df

    def get_urban_area(
        self,
        city: Union[int, str],
        area_factor: float,
        figsize: Tuple[int, int],
        min_area: int = 50,
    ):
        """
        This function tries zoom-in into the 'most important' urban area of the municipality
        Area factor is a value between 0 and 1 that specifies how much of the city must be included
        `min_area` is the minimum area in km2 to be considered in the output frame
        """

        # first, get the desired city
        city_df = self.get_city(city)

        # we will order by greatest areas to select just them
        city_df = city_df.sort_values(by="area_km2", ascending=False)
        city_df["cum_area"] = city_df["area_km2"].cumsum()

        # count how many items are required to have area greater than total_area * area_factor.
        iloc = (
            (~(city_df["cum_area"] > city_df["area_km2"].sum() * area_factor))
            .astype("int")
            .sum()
        )

        # get only those areas
        aoi_df = city_df.iloc[: iloc + 1]

        # once we have the urban areas that NEED to be considered, let's
        # get the extents to be analyzed, considering this AOI
        aspect = figsize[0] / figsize[1]
        xlim, ylim = calc_aspects_lims(aoi_df, aspect=aspect, percent_buffer=0.05)

        # get the bounds as a Shapely polygon
        bounds = gpd.GeoDataFrame(
            geometry=[box(xlim[0], ylim[0], xlim[1], ylim[1])]
        ).set_crs(self.crs)
        bounds = bounds.to_crs("esri:54034")

        if (bounds.area[0] * 1e-6) < min_area:
            # calculate the scale to be applied to the bounds
            scale = (min_area / (bounds.area[0] * 1e-6)) ** 0.5
            bounds = bounds.scale(scale, scale)

        # return bounds to original CRS
        bounds = bounds.to_crs(self.crs)

        aoi_df = city_df.clip(bounds)
        UrbanAreas.update_area(aoi_df)

        return UrbanArea(aoi_df, bounds)

    def iter_cities(self, area_factor: float, figsize: Tuple[int, int]) -> Iterable:
        """Return an iterable object to be used in a for loop"""
        uai = UrbanAreasIterator(self, area_factor, figsize)
        return uai

    @staticmethod
    def update_area(gdf: gpd.GeoDataFrame):
        """This static method update/create the 'area_km2' in the dataframe"""

        # calculate the area of each urban area in km2
        gdf_equal_area = gdf.to_crs("ESRI:54034")
        gdf["area_km2"] = gdf_equal_area.geometry.area / 1e6

    def __len__(self):
        """Return the number of urban areas"""
        return len(self.urban_areas["CD_MUN"].unique())


class UrbanAreasIterator:
    """Iterator through the cities"""

    def __init__(
        self,
        uas: UrbanAreas,
        area_factor: float,
        figsize: Tuple[int, int],
    ):
        self.idx = 0
        self.uas = uas
        # self.items = self.uas.urban_areas["CD_MUN"].astype("int").unique().tolist()

        areas = self.uas.urban_areas[["CD_MUN", "AREA_KM2"]].groupby(by="CD_MUN").sum()
        self.items = (
            areas.sort_values("AREA_KM2", ascending=False).index.astype("int").to_list()
        )

        self.area_factor = area_factor
        self.figsize = figsize

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.items):
            raise StopIteration

        ua = self.uas.get_urban_area(
            city=self.items[self.idx],
            area_factor=self.area_factor,
            figsize=self.figsize,
        )

        self.idx += 1
        return ua
