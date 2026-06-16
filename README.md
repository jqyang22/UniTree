
<div align="center">

<h1>Unified Knowledge Transfer Boost Individual Tree Crown Segmentation without Scene-specific Labels</h1>

<h2>Under Review</h2>
<h2>(The code will be made publicly available upon publication. For early access, please feel free to reach out.)</h2>


[Jiaqi Yang](https://jqyang22.github.io/)<sup>a</sup>, [Kyle Kabasares](https://www.kylekabasares.com/)<sup>b, c</sup>, [Ming Liu](https://pages.cs.wisc.edu/~mgliu/)<sup>d</sup>, [Taejin Park](https://www.nasa.gov/people/taejin-park/)<sup>b, c</sup>, [Min Chen](https://globalchange.cals.wisc.edu/staff/chen-min/)<sup>a, e ∗</sup>

<sup>a</sup> Department of Forest and Wildlife Ecology, University of Wisconsin-Madison, Madison, WI, USA,
<sup>b</sup> Bay Area Environmental Research Institute, Moffett Field, CA, 94035, USA,
<sup>c</sup> NASA Ames Research Center, Moffett Field, CA, 94035, USA, 
<sup>d</sup> Department of Computer Sciences, University of Wisconsin-Madison, 1210 W. Dayton Street, Madison, WI, 53706, USA, 
<sup>e</sup> Data Science Institute, University of Wisconsin-Madison, 447 Lorch Ct, Madison 53706 WI, USA.

<sup>*</sup> Corresponding author

</div>

<div align="center">

<!-- <p align='center'>
  <a href="ieeexplore.ieee.org/abstract/document/9321744"><img alt="Pape" src="https://img.shields.io/badge/RSE-Paper-6D4AFF?style=for-the-badge" /></a>
</p> -->


<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-study-areas">Study areas</a> |
  <!-- <a href="#-citation">Citation</a> | -->
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a> |
  <a href="#-acknowledgement">Acknowledgement</a>
</p >
</div>

<!-- 
<figure>
<div align="center">
<img src=Fig/logo1.png width="20%">
</div>
</figure>
-->



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
There are two distinct study areas in our study: [Denmark](https://sid.erda.dk/share_redirect/eFt21tspNe/denmark/extracted_data_train_patch_normalized_updated.zip) as the source domain and [Yosemite National Park](https://naip-usdaonline.hub.arcgis.com/) as the target domain.

<figure>
<div align="center">
<img src=Fig/Yosemite_national_park.bmp width="80%">
</div>
</figure>

<div align='center'>
 
**Figure 3. Test areas and target domains in Yosemite National Park.**

</div>

# 🔨 Usage

<!--
## Requirements
Python 3.9.20 and more in [environment.yml](environment.yml)
-->

## Train the model from scratch

--- 🔖 Set configs ---
config/Preprocessing.py

```
python main1-2_segcount_transfer_learning.py
```


<!--
# ⭐ Citation

If you find UniTree helpful, please give a ⭐ and cite it as follows:

```

```
-->

# 📒 Statement

For any other questions, please contact Jiaqi Yang at [jiaqi.yang@wisc.edu](mailto:jiaqi.yang@wisc.edu) or Min Chen at [min.chen@wisc.edu](mailto:min.chen@wisc.edu).


# 💖 Acknowledgement
The source-domain data is generated from [TreeCountSegHeight](https://github.com/sizhuoli/TreeCountSegHeight?tab=readme-ov-file). Thanks for their wonderful work!<br>

<!--
<img src="https://visitor-badge.laobi.icu/badge?page_id=WHU-Sigma.HyperSIGMA&left_color=%2363C7E6&right_color=%23CEE75F">

[![Star History Chart](https://api.star-history.com/svg?repos=WHU-Sigma/HyperSIGMA&type=Date)](https://star-history.com/#WHU-Sigma/HyperSIGMA&Date)
-->
