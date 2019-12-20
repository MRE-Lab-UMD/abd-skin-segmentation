# Deep Learning Techniques for Skin Segmentation on Novel Abdominal Dataset
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
---
<!--1) Add License file ( Change the abve tag depending upon the license we select)
2) Info about paper, add a gif
3) Citation links 
4) Info about the dataset 
5) How to download dataset
6) Hot to run models.-->

## Introduction
This repository provides codes for the skin segmentation methods investigated in \[1\], mainly Mask-RCNN, U-Net, a Fully Connected Network and a MATLAB script for thresholding. The algorithms were primarily developed to perform abdominal skin segmentation for trauma patients using RGB images, as part of an ongoing research work for developing an autonomous robot for trauma assessment \[[2](https://epubs.siam.org/doi/abs/10.1137/1.9781611975758.2)\]\[3\].

#### Robotic abdominal ultrasound system with the camera view of the abdominal region, and the corresponding segmented skin mask.
<p align="center">
  <img width="800" src="https://i.ibb.co/0GBK7BJ/1.png">
</p>

#### Sample pictures of the abdominal dataset depicting different skin color intensities and body complexions.
<p align="center">
  <img src="https://i.ibb.co/HK8VDdt/2.png" width="800">
</p>


#### Improved segmentation results with the use of our proposed Abdominal dataset using U-Net. From left to right in columns: original image, ground truth, segmentation with the Abdominal dataset, and segmentation without the Abdominal dataset.
<p align="center">
  <img src="https://i.ibb.co/6X51NsF/3.png" width="600">
</p>

## Information on Abdominal Skin Dataset
Summary on the paper's abdominal section (gender, skin color).

## Downloading Abdominal Skin Dataset
The complete skin datasets containing the original images along with their masks (which include HGR, TDSD, Schmugge, Pratheepan, VDM, SFA, FSD and our abdominal dataset) can be download from the following [link](https://drive.google.com/open?id=15FEX2rHemvQdAtuidh2qf2anOpHHFpzE). These datasets have been sorted to follow the same format, and can be readily run in the codes. If you're only interested in the abdominal dataset, you can download it from [here](https://drive.google.com/open?id=1j6owfRdf1UnH2wVqZuCbO9fY5-qaAiQm). You can also download and unzip the datasets from the terminal:
```
$ pip install gdown
$ gdown "https://drive.google.com/uc?id=15FEX2rHemvQdAtuidh2qf2anOpHHFpzE"
$ tar -xzvf  All_Skin_Datasets.tar.gz 
```
If you want to download the abdominal dataset separately:
```
$ gdown "https://drive.google.com/uc?id=1j6owfRdf1UnH2wVqZuCbO9fY5-qaAiQm"
$ tar -xzvf  Abdomen_Only_Dataset.tar.gz 
```

## Dependencies
Jupyter notebook, python 2(?), tenser flow, keras, opencv.

## Running the Codes
### Mask-RCNN
### U-Net
Summarize the comments in the code.
### Fully Connected Network
Again summarize the code.
### Thresholding
Clean up the MATLAB codes.

## Running the Models

## Citation
If you have used the abdominal dataset, or any of our trained models, kindly cite the associated paper:
```
@inproceedings{topiwala2019bibe,
    author = {A. Topiwala and L. Al-Zogbi and T. Fleiter and A. Krieger},
    title = {{Adaptation and Evaluation of Deep Leaning Techniques for Skin Segmentation on Novel Abdominal Dataset}},
  booktitle = {BIBE 2019; International Conference on Biological Information and Biomedical Engineering},
     pages = {1--8},
      year = {2019}
}
``` 

<!-- ## To Download the Datasets Used
```
pip install gdown
gdown  https://drive.google.com/uc?id=1CS_tl8cXOmy3Zv9TrqD2gApUTftKCu4m
```
To Unzip
```
tar -xzvf  Skin_Datasets.tar.gz 
```

 Feel Better :* -->
