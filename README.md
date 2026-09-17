
<div align="center">

<h1>Unified Knowledge Transfer Boosts Individual Tree Crown Segmentation without Scene-specific Labels</h1>

<h2><em>Remote Sensing of Environment (RSE)</em></h2>

[Jiaqi Yang](https://jqyang22.github.io/)<sup>a</sup>, [Kyle Kabasares](https://www.kylekabasares.com/)<sup>b, c</sup>, [Ming Liu](https://pages.cs.wisc.edu/~mgliu/)<sup>d</sup>, [Taejin Park](https://www.nasa.gov/people/taejin-park/)<sup>b, c</sup>, [Min Chen](https://globalchange.cals.wisc.edu/staff/chen-min/)<sup>a, e ∗</sup>

<sup>a</sup> Department of Forest and Wildlife Ecology, University of Wisconsin-Madison, Madison, WI, USA,
<sup>b</sup> Bay Area Environmental Research Institute, Moffett Field, CA, 94035, USA,
<sup>c</sup> NASA Ames Research Center, Moffett Field, CA, 94035, USA, 
<sup>d</sup> Department of Computer Sciences, University of Wisconsin-Madison, 1210 W. Dayton Street, Madison, WI, 53706, USA, 
<sup>e</sup> Data Science Institute, University of Wisconsin-Madison, 447 Lorch Ct, Madison 53706 WI, USA.

<sup>*</sup> Corresponding author

</div>

<div align="center">

<p align='center'>
<a href="https://doi.org/10.1016/j.rse.2026.115646">
  <img alt="Paper" src="https://img.shields.io/badge/RSE-Paper-orange?style=for-the-badge" />
</a>
</p>


<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-study-areas">Study areas</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-reference">Reference</a> |
  <a href="#-contact">Contact</a> |
  <a href="#-acknowledgement">Acknowledgement</a>
</p >
</div>


# 🌳 Overview

Accurate delineation of individual tree crowns is essential for forest inventory, ecosystem monitoring, and climate-informed management. However, most existing deep learning approaches rely on scene-specific annotations that are costly to produce and difficult to generalize across heterogeneous regions and image resolutions.</a>


<figure>
<div align="center">
<img src=Fig/Challenges.bmp width="80%">
</div>

<div align='center'>
 
**Figure 1. Challenges of individual tree segmentation.**

</div>
<br>

To tackle these challenges, we propose **UniTree**, a unified framework that enables cross-region and cross-resolution individual tree crown segmentation without scene-specific annotation.</a>

<figure>
<div align="center">
<img src=Fig/Overall_pipeline.bmp width="80%">
</div>

<div align='center'>
 
**Figure 2. Framework of UniTree.**

</div>
<br>


# 🌍 Study areas
There are two distinct study areas in our study: Denmark as the source domain and Yosemite National Park as the target domain.

Denmark data can be download from [Denmark](https://sid.erda.dk/share_redirect/eFt21tspNe/denmark/extracted_data_train_patch_normalized_updated.zip). Yosemite National Park data are extracted from [NAIP imagery](https://naip-usdaonline.hub.arcgis.com/).

<figure>
<div align="center">
<img src=Fig/Yosemite_national_park.bmp width="80%">
</div>
</figure>

<div align='center'>
 
**Figure 3. Test areas and target domains in Yosemite National Park.**

</div>


# 🔨 Usage

## Requirements
Python 3.9 and more in [environment.yml](environment.yml)

* Clone this repository and set environment

```
git clone https://github.com/jqyang22/UniTree.git
conda env create -f environment.yml
conda activate py39
```

* Please replace all file and directory paths with your local paths before running the code.

## Prepare your own data
**Source domain (labelled).** One flat folder. Each training frame is a set of single-band rasters that
share the same grid, named `<layer>_<id>.png`:

| layer | dtype | content |
|---|---|---|
| `red_<id>`, `green_<id>`, `blue_<id>`, `infrared_<id>` | float32 | image bands, already standardised per band (the training config runs with `normalize = 0`, so no normalisation is applied) |
| `annotation_<id>` | int16 | binary crown mask, {0, 1} |
| `boundary_<id>` | int16 | binary crown-boundary mask, {0, 1} |
| `ann_kernel_<id>` | float32 | Gaussian density map |

Frames are discovered by listing the files that start with `channel_names[0]`, so every frame needs all
seven layers. For three-band imagery, set `channel_names = ['red', 'green', 'blue']` in the training config.

**Target domain (unlabelled).** Same folder convention. Target imagery is what defines the domain you want to transfer to.

**Tiles for inference.** Georeferenced GeoTIFF, bands ordered as in `config.channels`,
about 0.6 m ground sampling distance.

**Reference data for evaluation.** A crown-polygon shapefile plus a rectangle shapefile delimiting the
annotated area.

## 1. Train the model
* Set configs <br>
[config/UniTreeTraining.py](config/UniTreeTraining.py)
* Run
```
python main1_train.py
```

## 2. Inference on new data
* Set configs <br>
[config/RasterAnalysis.py](config/RasterAnalysis.py)
* Run
```
python main2_infer.py
```

## 3. Evaluate
* Set configs <br>
[config/RasterAnalysis.py](config/RasterAnalysis.py)
* Run
```
python main3_eval.py \
    --pred  data/test_pred/<tile>_seg.tif \
    --label data/test/<tile>/lbl/<tile>_seg_polygon.shp \
    --rect  data/test/<tile>/lbl/<tile>_seg_rectangle.shp
```


# ⭐ Reference

For any questions, please feel free to reach me at [jiaqi.yang@wisc.edu](mailto:jiaqi.yang@wisc.edu)
If you find UniTree helpful, please give a ⭐ and cite it as follows:

```
@article{Yang2026Unified,
  title     = {Unified knowledge transfer boosts individual tree crown segmentation without scene-specific labels},
  author    = {Yang, Jiaqi and Kabasares, Kyle and Liu, Ming and Park, Taejin and Chen, Min},
  journal   = {Remote Sensing of Environment},
  volume    = {347},
  pages     = {115646},
  year      = {2026},
  issn      = {0034-4257},
  doi       = {10.1016/j.rse.2026.115646},
  publisher = {Elsevier}
}
```


# 📒 Contact

Jiaqi Yang: [jiaqi.yang@wisc.edu](mailto:jiaqi.yang@wisc.edu) <br>.
Min Chen: [min.chen@wisc.edu](mailto:min.chen@wisc.edu).


# 💖 Acknowledgement
The source-domain Denmark data is generated from [TreeCountSegHeight](https://github.com/sizhuoli/TreeCountSegHeight?tab=readme-ov-file). Thanks for their wonderful work!<br>
