"""
Implements the ProcessorReporter class
"""

from typing import Tuple, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from PyPDF2 import PdfMerger
import contextily as cx
from shapely import box

from .floodprocessor import FloodProcessor
from .utils import fig2pdf

# # the objective here is to save the figure, so we will be using the Agg backend
# current_backend = mpl.get_backend()
# mpl.use("agg")

# mpl.use(current_backend)


class ProcessorReporter:
    """Docstring"""

    def __init__(
        self,
        processor: FloodProcessor,
        title: str,
        flood_threshold: float = 1,  # 1km2 = 100ha
        aoi_df: Optional[gpd.GeoDataFrame] = None,
    ):
        self.processor = processor
        self.title = title
        self.aoi_df = aoi_df
        self.flood_threshold = flood_threshold

    def create_context_page(self, figsize: Tuple[int, int]):
        """Create the page with the context for the place
        Args:
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Figure to be saved into the PDF
        """

        # change Matplotlib backend
        current_backend = mpl.get_backend()
        mpl.use("agg")

        fig, axs = plt.subplots(2, 1, figsize=(figsize[0], 2 * figsize[1]))

        title = f"Report: {self.title}"
        fig.suptitle(title)

        # plot the first image
        self.processor["aoi_df"].plot(
            ax=axs[0], facecolor="none", edgecolor="red", rasterized=True
        )

        # Add the context map
        axs[0].set_title("Context Map (ESRI satellite imagery as backcround)")
        cx.add_basemap(
            ax=axs[0], crs="epsg:4326", source=cx.providers.Esri.WorldImagery
        )

        gsw = self.processor["recurrence"]
        plot = gsw.squeeze().plot(
            ax=axs[1], cmap="YlGnBu", add_colorbar=False, vmin=10, vmax=100, zorder=1
        )
        self.processor["aoi_df"].plot(
            facecolor="grey",
            edgecolor="black",
            ax=axs[1],
            alpha=0.3,
            zorder=0,
            rasterized=True,
        )
        axs[1].set_title("Water Recurrence")
        axs[1].set_aspect(1.0)

        # Create a colorbar axis below the main plot
        cbar_ax = fig.add_axes(
            [0.5 - (0.35 / 2), 0.05, 0.35, 0.01]
        )  # Adjust the position and size as needed

        # Add the colorbar to the colorbar axis
        cbar = plt.colorbar(plot, cax=cbar_ax, orientation="horizontal")

        # Customize colorbar properties as needed
        cbar.set_label("Water Recurrence (%)")  # Replace with your label

        # plot the area of interest, if available
        if self.aoi_df is not None:
            self.aoi_df.plot(ax=axs[0], facecolor="none", edgecolor="white")
            self.aoi_df.plot(ax=axs[1], facecolor="red", edgecolor="red", alpha=0.2)

        mpl.use(current_backend)
        return fig

    def create_summary_table(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """

        df = self.processor["data_table"]

        num_floods = (df["Flood area"] > self.flood_threshold).sum()

        # if an area of interest is provided, calculate its area in km^2
        if self.aoi_df is not None:
            urban_area = self.aoi_df.to_crs("ESRI:54034").area.sum() * 1e-6
            urban_area = f"{urban_area:.2f}"

            # additionally, calculate the portion of the AOI that was flooded
            # the value is converted to ha
            max_flood = self.processor["max_flood"].rio.set_crs("epsg:4326")
            urban_flooded = max_flood.rio.clip(self.aoi_df.geometry)
            urban_flooded = float(urban_flooded.sum()) * 30 * 30 * 1e-6 * 100
            urban_flooded = f"{urban_flooded:.2f}"

        else:
            urban_area = " - "
            urban_flooded = " - "

        total_area = (
            box(*self.processor["aoi_df"].to_crs("esri:54034").total_bounds).area * 1e-6
        )

        # if "vulnerable" in self.processor.vars:
        #     vulnerable_area = int(self.processor["vulnerable"].sum()) * 900 * 1e-6
        #     urban_vul = int(self.processor["urban_vul"].sum()) * 900 * 1e-6
        # else:
        #     vulnerable_area = 0
        #     urban_vul = 0

        data = {
            "Localidade": self.processor.name,
            "Período": self.processor.finder.dates_range,
            "Imagens disponíveis": len(self.processor.finder.s1imagery),
            "Area urbana (km^2)": urban_area,
            "Area total monitorada (km^2)": f"{total_area:.2f}",
            "Limiar inundação (ha)": f"{self.flood_threshold * 100}",
            "Inundações detectadas": str(num_floods),
            # "Area vulneravel (km^2)": f"{vulnerable_area:.2f}",
            "Área urbana Inundada (ha)": urban_flooded,
            "Máxima cheia: ": f"{df.index.astype('str')[df['Water Extents'].argmax()]}",
        }

        df = pd.Series(data).to_frame()
        df.to_csv(self.processor.output_dir / "summary.csv")
        return df

    def plot_summary_table(self, ax: plt.Axes):
        """
        Plot the summary table into a specific Axes
        Args:
            ax (plt.Axes): Axes to plot the table
        """
        # Hide the axes
        ax.axis("off")

        df = self.create_summary_table().reset_index()

        # Plot the table
        # table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table = ax.table(cellText=df.values, cellLoc="center", loc="center")

        # You can customize the appearance of the table if needed
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        ax.set_title("Resumo")

    def create_summary_page(self) -> plt.Figure:
        """Docstring"""

        # change Matplotlib backend
        current_backend = mpl.get_backend()
        mpl.use("agg")

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        self.plot_summary_table(ax=axs[0])
        self.processor["data_table"]["Flood area"].plot(ax=axs[1])
        axs[1].set_title("Flooded Area (km^2)")

        xmin, xmax, _, _ = axs[1].axis()
        axs[1].hlines(y=self.flood_threshold, xmin=xmin, xmax=xmax, colors="red")

        mpl.use(current_backend)

        return fig

    def create_s1_page(self) -> plt.Figure:
        """Create the PDF page with S1 imagery"""

        # change Matplotlib backend
        current_backend = mpl.get_backend()
        mpl.use("agg")

        # Get dates for high filling and low filling
        ws = self.processor["water_series"].sort_values()
        low_filling_date = ws[ws <= ws.quantile(0.1)].index.astype("str")[-1]
        # high_filling_date = ws[ws < ws.quantile(0.95)].index.astype("str")[-1]
        high_filling_date = ws.index.astype(str)[-1]

        # plot the images
        fig, axs = plt.subplots(2, 1, figsize=(12, 18))
        fig.suptitle("Imagens S1 - Terrain Corrected")

        self.processor.finder.s1imagery.plot_date(low_filling_date, raw=True, ax=axs[0])
        self.processor.plot_var("aoi_df", ax=axs[0], facecolor="none", edgecolor="red")
        axs[0].set_title(f"Menor superfície de água: {low_filling_date}")

        self.processor.finder.s1imagery.plot_date(
            high_filling_date, raw=True, ax=axs[1]
        )
        self.processor.plot_var("aoi_df", ax=axs[1], facecolor="none", edgecolor="red")
        axs[1].set_title(f"Maior superfície de água: {high_filling_date}")

        # plot the area of interest, if available
        if self.aoi_df is not None:
            self.aoi_df.plot(ax=axs[0], facecolor="none", edgecolor="white")
            self.aoi_df.plot(ax=axs[1], facecolor="none", edgecolor="white")

        mpl.use(current_backend)

        return fig

    def create_flood_page(self) -> plt.Figure:
        """Create the PDF page with the inundation maps"""

        # change Matplotlib backend
        current_backend = mpl.get_backend()
        mpl.use("agg")

        fig, axs = plt.subplots(2, 1, figsize=(10, 17))

        # Plot water recurrence
        self.processor.plot_var(
            "aoi_df", facecolor="none", ax=axs[0], edgecolor="black"
        )

        self.processor.plot_var(
            "recurrence", ax=axs[0], cmap="Blues", add_colorbar=False
        )
        cx.add_basemap(
            ax=axs[0], crs="epsg:4326", source=cx.providers.Esri.WorldImagery
        )

        # Plot Max Flood
        self.processor.plot_var(
            "recurrence", ax=axs[1], cmap="Blues", add_colorbar=False
        )
        self.processor.plot_var("max_flood", ax=axs[1], add_colorbar=False, cmap="Reds")
        self.processor.plot_var(
            "aoi_df", facecolor="none", ax=axs[1], edgecolor="black"
        )
        cx.add_basemap(
            ax=axs[1], crs="epsg:4326", source=cx.providers.Esri.WorldImagery
        )

        # if "extrapolated_flood" in self.processor.vars:
        #     self.processor.plot_extrapolated_flood(ax=axs[1])
        #     # processor.plot_var('max_flood', ax=axs[1], add_colorbar=False, cmap='Reds')

        axs[0].set_title("Water Recurrence")
        axs[1].set_title("Water Recurrence and maximum detected flood (Red)")

        # plot the area of interest, if available
        if self.aoi_df is not None:
            self.aoi_df.plot(ax=axs[0], facecolor="none", edgecolor="white")
            self.aoi_df.plot(ax=axs[1], facecolor="none", edgecolor="white")

        mpl.use(current_backend)

        return fig

    def create_all_floods_page(self) -> plt.Figure:
        """Create the PDF page with the various floods (up to 8)"""

        # first of all get all detected floods
        floods = self.processor["data_table"]["Flood area"] > self.flood_threshold
        floods = self.processor["data_table"][floods]

        # then, order by flooded ammount
        floods = floods.sort_values(["Flood area"], ascending=False)

        # if at least 2 floods, call the function
        if len(floods) > 1:

            # change Matplotlib backend
            current_backend = mpl.get_backend()
            mpl.use("agg")

            fig, axs = plt.subplots(4, 2, figsize=(12, 25))

            for i, date in enumerate(floods.index):
                if i >= 8:
                    break
                self.processor.plot_flood(
                    ax=axs.reshape(-1)[i],
                    date=date.strftime("%Y-%m-%d"),
                    background=cx.providers.Esri.WorldImagery,
                )

                # plot the area of interest, if available
                if self.aoi_df is not None:
                    self.aoi_df.plot(
                        ax=axs.reshape(-1)[i], facecolor="none", edgecolor="white"
                    )

            mpl.use(current_backend)

            return fig
        else:
            return None

    def create_dem_page(self) -> plt.Figure:
        """Create the PDF page with DEM and HAND to be appended for the report"""

        # create memory-like objects to store the PNG and PDF
        # png = io.BytesIO()
        # pdf = io.BytesIO()

        fig, axs = plt.subplots(2, 1, figsize=(12, 23))

        self.processor.plot_var("dem", ax=axs[0], add_colorbar=False)
        self.processor.plot_var(
            "aoi_df", ax=axs[0], facecolor="none", edgecolor="white"
        )
        axs[0].set_title("ANADEM Model")

        self.processor.plot_var("hand", ax=axs[1], add_colorbar=False)
        self.processor.plot_var(
            "aoi_df", ax=axs[1], facecolor="none", edgecolor="white"
        )
        axs[1].set_title("HAND Model")

        # self.plot_vars(ax=ax[1], dem="hand")

        return fig

        # fig.savefig(png, dpi=150, format="png")
        # png.seek(0)
        # pdf.write(img2pdf.convert(png))  # type: ignore

        # pdf.seek(0)
        # plt.close(fig)

        # # return the original backend

        # return pdf

    def create_report(self):
        """Create the report and save it to the output directory"""

        # change the backend to "agg" to avoid memory leakage
        current_backend = mpl.get_backend()
        mpl.use("agg")

        pdf_merger = PdfMerger()

        fig = self.create_context_page(figsize=(8, 6))
        pdf_merger.append(fig2pdf(fig))

        fig = self.create_summary_page()
        pdf_merger.append(fig2pdf(fig))

        # fig = self.create_dem_page()
        # pdf_merger.append(fig2pdf(fig))

        fig = self.create_s1_page()
        pdf_merger.append(fig2pdf(fig))

        fig = self.create_flood_page()
        pdf_merger.append(fig2pdf(fig))

        fig = self.create_all_floods_page()
        if fig is not None:
            pdf_merger.append(fig2pdf(fig))

        mpl.use(current_backend)

        pdf_merger.write(self.processor.output_dir / "report.pdf")
