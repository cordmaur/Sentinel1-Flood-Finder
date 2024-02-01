"""Docstring"""
import io
from pathlib import Path
from typing import Tuple, Union
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

import xarray as xr
import rioxarray as xrio

import img2pdf


def fig2pdf(fig: plt.Figure, dpi: int = 150):
    """Convert a matplotlib figure to a file-like object (in-memory) PDF.

    Args:
        fig (plt.Figure): _description_
        dpi (int, optional): _description_. Defaults to 150.
    """

    # create memory objects
    pdf = io.BytesIO()
    png = io.BytesIO()

    # first, let's save the figure as an image
    fig.savefig(png, format="jpg", dpi=dpi, transparent=False)

    # convert it to PDF
    png.seek(0)
    pdf.write(img2pdf.convert(png))

    # clear the png
    png.seek(0)
    png.truncate(0)

    # return the PDF
    pdf.seek(0)
    return pdf


def adjust_coords(arr: xr.DataArray, ref_arr: xr.DataArray) -> xr.DataArray:
    """
    Make sure the array has the same coords as the reference array
    Args:
        arr (xr.DataArray): DataArray to be ajusted
        ref_arr (xr.DataArray): DataArray with the reference coords

    Returns:
        xr.DataArray: DataArray with the same coordinates as the reference array
    """

    if (len(arr.x) != len(ref_arr.x)) or (len(arr.y) != len(ref_arr.y)):
        raise ValueError("Adjusting coords requires the arrays to have the same shape")

    arr = arr.assign_coords({"x": ref_arr.x, "y": ref_arr.y})

    return arr


def open_tif_as_dset(tif: Union[str, Path], nodata: int = 0) -> xr.Dataset:
    """Open a .tif file and convert it to a Dataset"""
    arr = xrio.open_rasterio(tif)
    dset = arr.to_dataset(dim="band")

    if len(dset) > 1:
        dset = dset.rename_vars({i + 1: date for i, date in enumerate(dset.long_name)})
    else:
        dset = dset.rename_vars({1: dset.long_name})

    del dset.attrs["long_name"]

    for var in dset:
        dset[var].rio.write_nodata(nodata, inplace=True)

    return dset


def calc_bounds(
    shp: gpd.GeoDataFrame, percent_buffer: float = 0, fixed_buffer: float = 0.0
) -> tuple:
    """
    Return the total bounds of a shape file with a given buffer
    The buffer can be a fixed distance (in projection units)
    or a percentage of the maximum size
    """

    # get the bounding box of the total shape
    bbox = box(*shp.total_bounds)

    if fixed_buffer != 0:
        bbox = bbox.buffer(fixed_buffer)
    elif percent_buffer != 0:
        xmin, ymin, xmax, ymax = bbox.bounds
        delta_x = xmax - xmin
        delta_y = ymax - ymin
        diag = (delta_x**2 + delta_y**2) ** 0.5
        bbox = bbox.buffer(percent_buffer * diag)

    return bbox.bounds


def calc_aspects_lims(
    shp: gpd.GeoDataFrame,
    aspect: float = 1.0,
    percent_buffer: float = 0,
    fixed_buffer: float = 0.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the limits of a viewport given a GeoDataFrame, a buffer around the bounds (optional)
    and an aspect ratio: aspect_ratio = lim_x/lim_y
    the buffer can be expressed as crs units or percentage
    """

    # first, let's get the bounding box
    xmin, ymin, xmax, ymax = calc_bounds(
        shp=shp, percent_buffer=percent_buffer, fixed_buffer=fixed_buffer
    )

    # calc the sizes in each dimension
    size_x = xmax - xmin
    size_y = ymax - ymin

    actual_aspect = size_x / size_y

    # if actual aspect is smaller, that means width has to be increased
    if actual_aspect < aspect:
        # we have to increase X accordingly
        delta = size_y * aspect - size_x
        xmin -= delta / 2
        xmax += delta / 2

    # if actual aspect is greater, that means height has to be increased
    else:
        # we have to increase Y axis accordingly
        delta = size_x / aspect - size_y
        ymin -= delta / 2
        ymax += delta / 2

    # return the limits
    return (xmin, xmax), (ymin, ymax)
