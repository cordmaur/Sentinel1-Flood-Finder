# Sentinel1 Flood Finder
### ML Flood Finder Package from Sentinel 1 (SAR) Imagery

![image](https://github.com/cordmaur/Sentinel1-Flood-Finder/assets/19617404/f2e69526-1ae1-48b4-85e8-181abd012a7a)


## Introduction
Floods, among the most devastating natural hazards, impact the lives of millions each year, causing tragic losses and widespread destruction. Timely forecasts and alert systems are crucial to minimizing deaths and damage, and obtaining accurate floodwater extent measurements is vital for effective emergency response. While traditional field surveys can be costly and impractical, remote sensing offers a powerful solution.

However, conventional water detection techniques relying on optical satellite imagery often face limitations during floods. Intense cloud cover can obscure the ground, rendering optical sensors ineffective. In such scenarios, radar technology (Synthetic Aperture Radar, SAR) offers a critical advantage, being able to penetrate clouds and capture valuable data. However, water mapping using SAR imagery is not straightforward and requires specialized techniques and tools.

## Introducing S1FloodFinder: A User-Friendly Solution for Flood Mapping

To address this challenge, the Python package `S1FloodFinder` was developed as a user-friendly solution for generating flood maps directly from Sentinel-1 SAR imagery. The package leverages radiometrically corrected S1 data readily available on the Microsoft Planetary Computer (Figure 1). This cloud-native approach eliminates the need for users to download and manage large volumes of imagery, streamlining the process significantly.

<img src="https://github.com/cordmaur/Sentinel1-Flood-Finder/assets/19617404/47e2d803-05fa-47cc-8a9f-85e1a1a71803" width="500" height="300" alt="Image description">
<div align="left">Figure 1: Example of Sentinel 1 RTC image from MS Planetary Computer. Image from Porto Alegre - RS region.</div>


### Machine Learning Methodology

The `S1FloodFinder` utilizes a pixel-based Random Forests approach on the `VV` and `VH` polarizations to detect water pixels within a scene. The model was trained using the CNES ALCD Open water masks [1]. The water masks cover distinct regions and weather seasons over France and were originally obtained from the Sentinel 2 imagery. During training phase, these masks were paired to the Sentinel 1 imagery (Figure 2). The pre-trained model is available on the `/model/RF_Model_v2.joblib` file.

![image](https://github.com/cordmaur/Sentinel1-Flood-Finder/assets/19617404/d7870222-1065-432c-b4e2-70dc255a901d)
<div align="center">Figure 2: Example of S1, S2 and water masks  pairing.</div>

### Water extents baseline
An additional step in delineating floodwater extent involves identifying permanent and recurrent water baselines. For this purpose, we employ the Global Water Surface dataset (Pekel et al., 2016) [2], also available from the Planetary Computer. Permanent water bodies are dynamic, so the goal is to prevent areas with recurring inundation from being classified as flood zones. A threshold is applied to the <b>Recurrency</b> layer, which accounts for the inter-annual recurrence of water pixels. Figure 3 illustrates the distinction between the more common <b>Water Occurrence</b> layer and the <b>Water Recurrence</b> layer, focusing on Eldorado du Sul - RS, where numerous rice crop farms exist. These areas experience seasonal inundation. Despite their limited inundation period (low occurrence), we aim to remove them from the final flood extent map, hence the use of the water recurrence layer.

<img src="https://github.com/cordmaur/Sentinel1-Flood-Finder/assets/19617404/f0767b4f-a56b-43e1-ba14-c3ddb73edd7c" width="1000" height="600" alt="Image description">
<div align="center">Figure 3: GLobal Surface Water Occurrence and Recurrence layers over Eldorado do Sul - RS rice crops. .</div>

### Results Examples
The package can create flood maps automatically, without the hassle of image downloading or pre-processing. Figure 4 shows an example of the outputs for Alegrete-RS/Brazil. The time series shows the result of the flood detection for the whole period of imagery available from Sentinel 1, from 2016 to the present. We can notice the major flood occurred on January 2019. These outputs are created automatically and it is explained in the notebooks available in `nbs`. 

<img src="https://github.com/cordmaur/Sentinel1-Flood-Finder/assets/19617404/aada58ee-b128-44ae-845e-de599ed64552" width="900" height="500" alt="Image description">
<div align="center">Figure 4: Results for Alegrete-RS Jan/2019 flood.</div>


## Prerequisites
These packages are necessary to run `S1FloodFinder`:
```
gdal
geopandas
rioxarray 
pystac-client 
matplotlib
pytest
cfgrib
netCDF4 
notebook 
contextily 
eccodes 
adjustText
pyarrow 
unidecode 
ipywidgets 
stackstac 
tqdm 
scikit-image==1.2.2
pypdf2
img2pdf
planetary-computer
pystac
```

Scikit version is pinned to `1.2.2` to guarantee compatibility with the pre-trained Random Forests model. 

If you are familiar with docker and prefer conteinerized development, the following command pulls the docker image with all the packages pre-installed:
```
docker pull cordmaur/planetary:v1
```

<i><b>PS:</b> To learn more about the use of docker for geospatial development, I have a series of posts about this subject on <b>GeoCorner (http://geocorner.net)</b>:
* [Why You Should Use Devcontainers for Your Geospatial Development](https://www.geocorner.net/post/why-you-should-use-devcontainers-for-your-geospatial-development)
* [Configuring a Minimal Docker Image for Spatial Analysis with Python](https://www.geocorner.net/post/configuring-a-minimal-docker-image-for-spatial-analysis-with-python)
* [Don't Install Python (Locally) for Data Science. Use Docker Instead!](https://www.geocorner.net/post/don-t-install-python-for-data-science-use-docker-instead)
</i>


## Instalation
This package is not yet available through `PyPI`, so the installation process can be done by pulling the package directly from github, like so:

`pip install git+https://github.com/cordmaur/Sentinel1-Flood-Finder.git@main`

## Usage
For package usage, please refer to the file `nbs/Introduction.ipynb`.


## References
[1] PENA LUQUE Santiago. (2019). CNES ALCD Open water masks (1.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3522069

[2] Pekel, JF., Cottam, A., Gorelick, N. et al. High-resolution mapping of global surface water and its long-term changes. Nature 540, 418–422 (2016). https://doi.org/10.1038/nature20584
