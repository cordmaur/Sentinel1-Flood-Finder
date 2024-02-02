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

The `S1FloodFinder` uses a pixel-based Random Forests approach on the `VV` and `VH` polarizations to detect water pixels within a scene. The model was trained using the CNES ALCD Open water masks [1]. The water masks cover distinct regions and weather seasons over France and were originally obtained from the Sentinel 2 imagery. These masks were paired to the Sentinel 1 imagery (Figure 2) for training purposes.

![image](https://github.com/cordmaur/Sentinel1-Flood-Finder/assets/19617404/d7870222-1065-432c-b4e2-70dc255a901d)
<div align="center">Figure 2: Example of S1, S2 and water masks  pairing.</div>

### Water extents baseline
One additional step when delineating water flood extension 

## References
[1] PENA LUQUE Santiago. (2019). CNES ALCD Open water masks (1.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3522069
![image](https://github.com/cordmaur/Sentinel1-Flood-Finder/assets/19617404/285b7561-5443-476a-b641-81ae73a301c8)
