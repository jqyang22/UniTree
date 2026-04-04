
<div align="center">

<h1>Unified Knowledge Transfer Boost Individual Tree Crown Segmentation without Scene-specific Labels</h1>

<h2>Under Review</h2>


[Jiaqi Yang](https://jqyang22.github.io/)<sup>a</sup>, [Kyle Kabasares](https://www.kylekabasares.com/)<sup>b, c</sup>, [Ming Liu](https://pages.cs.wisc.edu/~mgliu/)<sup>d</sup>, [Taejin Park](nasa.gov/people/taejin-park/)<sup>b, c</sup>, [Min Chen](https://globalchange.cals.wisc.edu/staff/chen-min/)<sup>a, e ∗</sup>

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
  <a href="#-citation">Citation</a> |
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




<!-- # 🔥 Update

**2025.11.24**

- **HyperGlobal-450K** can be assesed through **<a href="https://pan.baidu.com/s/1duYGTpeEcuQkLByTSjyOtQ?pwd=j9pv">Baidu Drive (百度网盘) <img height="15" width="15" src="https://cdn.jsdelivr.net/npm/simple-icons@v13/icons/baidu.svg"/></a>**.

**2025.11.15**

- 🏆 HyperSIGMA is selected as a Highly Cited Paper!

**2025.04.08**

- The main paper is online published! Please see **[here](https://ieeexplore.ieee.org/document/10949864)**.

**2025.04.02**

- The **[arXiv](https://arxiv.org/abs/2406.11519)** is updated with more details. Stay tuned for further updates!
- **HyperGlobal-450K** has been released! Please refer to **[here](https://huggingface.co/datasets/WHU-Sigma/HyperGlobal-450K)**.

**2025.03.31**

- We are delighted to annouce **HyperSIGMA** has been accepted by **IEEE TPAMI**!

**2024.10.22**
- Scripts for **[Image Super-Resolution](https://huggingface.co/datasets/WHU-Sigma/HyperSIGMA_Datasets/tree/main/HyperSIGMA_super-resolution)**.

- Checkpoints for **[Image Denoising](https://huggingface.co/WHU-Sigma/HyperSIGMA/tree/main/Denoising_models)**.

**2024.07.18**
- Models can be downloaded from both **<a href="#-pretrained-models">Baidu Drive (百度网盘) <img height="15" width="15" src="https://cdn.jsdelivr.net/npm/simple-icons@v13/icons/baidu.svg"/></a>** and **[Hugging Face 🤗](https://huggingface.co/WHU-Sigma/HyperSIGMA/tree/main)**.

- Datasets for HSI denoising have been released for research use only. Please check it **[here](https://huggingface.co/datasets/WHU-Sigma/HyperSIGMA_Datasets/tree/main/HyperSIGMA_denoising)**.

**2024.06.18**
- The paper is post on arXiv! **([arXiv 2406.11519](https://arxiv.org/abs/2406.11519))** 
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

<!--
# 🚀 Pretrained Models

| Pretrain | Backbone | Model Weights |
| :------- | :------: | :------: |
| Spatial_MAE | ViT-B | [Baidu Drive](https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc) & [Hugging Face](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spat-vit-base-ultra-checkpoint-1599.pth)| 
| Spatial_MAE | ViT-L |  [Baidu Drive](https://pan.baidu.com/s/11iwHFh8sfg9S-inxOYtJlA?pwd=d2qs) & [Hugging Face](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spat-vit-large-ultra-checkpoint-1599.pth)|
| Spatial_MAE | ViT-H | [Baidu Drive](https://pan.baidu.com/s/1gV9A_XmTCBRw90zjSt90ZQ?pwd=knuu) & [Hugging Face](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spat-vit-huge-ultra-checkpoint-1599.pth)| 
| Spectral_MAE | ViT-B |  [Baidu Drive](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y) & [Hugging Face](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spec-vit-base-ultra-checkpoint-1599.pth)|
| Spectral_MAE | ViT-L | [Baidu Drive](https://pan.baidu.com/s/1tF2rG-T_65QA3UaG4K9Lhg?pwd=xvdd) & [Hugging Face](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spec-vit-large-ultra-checkpoint-1599.pth)| 
| Spectral_MAE | ViT-H |  [Baidu Drive](https://pan.baidu.com/s/1Di9ffWuzxPZUagBCU4Px2w?pwd=bi9r) & [Hugging Face](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spec-vit-huge-ultra-checkpoint-1599.pth) |
-->



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
## Finetuning

***Image Classification***: 

Please refer to [ImageClassification-README](https://github.com/WHU-Sigma/HyperSIGMA/tree/main/ImageClassification).

***Target Detection & Anomaly Detection***: 

Please refer to [HyperspectralDetection-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/main/HyperspectralDetection).

***Change Detection***: 

Please refer to [ChangeDetection-README](https://github.com/WHU-Sigma/HyperSIGMA/tree/main/ChangeDetection).


***Spectral Unmixing***: 

Please refer to [HyperspectralUnmixing-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/main/HyperspectralUnmixing).
-->


<!--
# ⭐ Citation

If you find UniTree helpful, please give a ⭐ and cite it as follows:

```
@ARTICLE{hypersigma,
  author={Wang, Di and Hu, Meiqi and Jin, Yao and Miao, Yuchun and Yang, Jiaqi and Xu, Yichu and Qin, Xiaolei and Ma, Jiaqi and Sun, Lingyu and Li, Chenxing and Fu, Chuan and Chen, Hongruixuan and Han, Chengxi and Yokoya, Naoto and Zhang, Jing and Xu, Minqiang and Liu, Lin and Zhang, Lefei and Wu, Chen and Du, Bo and Tao, Dacheng and Zhang, Liangpei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model}, 
  year={2025},
  volume={47},
  number={8},
  pages={6427-6444},
  keywords={Hyperspectral imaging;Foundation models;Transformers;Feature extraction;Training;Computer vision;Satellites;Computational modeling;Cameras;Scalability;Remote sensing;hyperspectral image;foundation model;attention;vision transformer;large-scale dataset},
  doi={10.1109/TPAMI.2025.3557581}
}
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
