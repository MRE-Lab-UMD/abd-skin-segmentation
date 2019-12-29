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

---

#### Robotic abdominal ultrasound system with the camera view of the abdominal region, and the corresponding segmented skin mask.
<p align="center">
  <img width="600" src="https://i.ibb.co/0GBK7BJ/1.png">
</p>

---

## Information on Abdominal Skin Dataset
The dataset consists of 1,400 abdomen images retrieved online from Google images search, which were subsequently manually segmented. The images were selected to preserve the diversity of different ethnic groups, preventing indirect racial biases in segmentation algorithms; 700 images represent darker skinned people, which include African, Indian and Hispanic groups, and 700 images represent lighter skinned people, such as Caucasian and Asian groups. A total of 400 images were selected to represent people with higher body mass indices, split equally between light and dark categories. Variations between individuals, such as hair and tattoo coverage, in addition to externals variations like shadows, were also accounted for in the dataset preparation. The size of the images is 227x227 pixels. The skin pixels form 66% of the entire pixel data, with a mean of 54.42% per individual image, and a corresponding standard deviation of 15%. 

---

#### Sample pictures of the abdominal dataset depicting different skin color intensities and body complexions.
<p align="center">
  <img src="https://i.ibb.co/HK8VDdt/2.png" width="600">
</p>

---

#### Improved segmentation results with the use of our proposed Abdominal dataset using U-Net. From left to right in columns: original image, ground truth, segmentation with the Abdominal dataset, and segmentation without the Abdominal dataset.
<p align="center">
  <img src="https://i.ibb.co/6X51NsF/3.png" width="400">
</p>

---

## Downloading Skin Datasets
The complete skin datasets containing the original images along with their masks (which include HGR, TDSD, Schmugge, Pratheepan, VDM, SFA, FSD and our abdominal dataset) can be download from the following [link](https://drive.google.com/open?id=15FEX2rHemvQdAtuidh2qf2anOpHHFpzE). These datasets have been sorted to follow the same format, and can be readily run in the codes. If you're only interested in the abdominal dataset, you can download it from [here](https://drive.google.com/open?id=1j6owfRdf1UnH2wVqZuCbO9fY5-qaAiQm). You can also download and unzip the datasets from the terminal:
```
$ pip install gdown
$ gdown "https://drive.google.com/uc?id=1xzYn4Rat4z2LA5zQW7JTfvA1bosz7oM-"
$ tar -xzvf All_Skin_Datasets.tar.gz 
```
If you want to download the abdominal dataset separately:
```
$ gdown "https://drive.google.com/uc?id=1MnBW_OJqrTmzwc23YI5NK_y_l4zk9JGJ"
$ tar -xzvf Abdomen_Only_Dataset.tar.gz 
```
The folders are organized as follows:
```
/All_Skin_Datasets/
             ├── Dataset1_HGR/
             │   ├── original_images/
             │   │     ├ <uniqueName1>.jpg
             │   │     .
             │   │     └ <uniqueNameK>.jpg
             |   └── skin_masks/
             |         ├ <uniqueName1>.png
             |         .
             |         └ <uniqueNameK>.png
             ├── Dataset2_TDSD/
             ├── Dataset3_Schmugge/
             .
             .
             └── Dataset8_Abdomen/
                 ├── test/            
                 |     ├── original_images/
                 │     |     ├ <uniqueName1>.jpg
                 │     |     .
                 │     |     └ <uniqueNameK>.jpg
                 │     └── skin_masks/
                 │           ├ <uniqueName1>.png
                 │           .
                 │           └ <uniqueNameK>.png
                 └── train/            
                       ├── original_images/
                       |     ├ <uniqueName1>.jpg
                       |     .
                       |     └ <uniqueNameK>.jpg
                       └── skin_masks/
                             ├ <uniqueName1>.png
                             .
                             └ <uniqueNameK>.png
```

## Dependencies
The codes require Python 3 to run. For installation run:
```
$ sudo apt-get update
$ sudo apt-get install python3.6
```
U-Net and the Fully Connected Network are written in Jupyter Notebook, so if you wish to run them, you should have it installed:
```
$ pip install jupyterlab
```
Next you need to install Tensorflow and Keras (it's better to install tenserflow gpu, otherwise it will take days to train your networks). The following steps include the installation of needed dependencies for this step:
```
$ pip install --upgrade tensorflow
$ pip install numpy scipy
$ pip install scikit-learn
$ pip install Pillow
$ pip install h5py
$ pip install keras
```
Some other dependencies for running the codes which are not included in the Python library:
```
$ pip install six matplotlib scikit-image opencv-python imageio Shapely
$ pip install imgaug
$ pip install talos
$ pip install tqdm
$ pip install Cython
$ pip install more-itertools
```
Finally clone the repository:
```
$ git clone --recursive https://github.com/MRE-Lab-UMD/abd-skin-segmentation.git
```

## Running the Codes
### Mask-RCNN

### U-Net
The U-Net notebook in the folder ```UNET and Features``` provides clear instructions and comments on each section and subsection. Just follow the guidelines to train your own network, and make sure you replace our paths with yours. The code will automatically save your model as .h5, which you can subsequently load for further usage. The notebook U-Net - Parameter Optimization contains the same code as U-Net, but trains the network over a set of hyperparameters to find the optimal ones. 

### Fully Connected Network
The Features notebook in the folder ```UNET and Features``` provides clear instructions and comments on each section and subsection. Just follow the guidelines to train your own network, and make sure you replace our paths with yours. The code will automatically save your model as .h5, which you can subsequently load for further usage. We recommend you read the entire instructions once before running any sections, as some of them will take a while to complete, so you want to make sure you're running the parts that are needed for you.

### Thresholding
Clean up the MATLAB codes.

### Real-Time Skin Segmentation
To run the real-time segmentation using our trained U-Net model, go to the ```Real Time Skin Segmentation``` directory in your terminal, and just type in:
```
$ python UNET-live.py
```
Make sure that you have set up your path to the trained model correctly in the code, and installed all required dependencies. Press on the ESC key to stop the code and close the camera window.
 
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

### Further Help and Suggestions
We can't guarantee that the codes will run perfectly on your machine (they should, but you never know). If you have any problems, questions, or suggestions, please feel free to contact the authors by email, we are pretty responsive and friendly. 
* Anirudh Topiwala: topiwala.anirudh@gmail.com
* Lydia Zoghbi: lalzogbi@terpmail.umd.edu

We hope to bring the best to the community! Cheers :heart::beers:!
