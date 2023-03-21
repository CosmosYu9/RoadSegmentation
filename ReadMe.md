# 2-Stage RoadSegmentation Approach

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This repo aims at implementing the 2-stage road segmentation approach proposed in [1] T. Li, M. Comer and J. Zerubia, "A Two-Stage Road Segmentation Approach for Remote Sensing Images, " ICPR2022 workshop

Note: the first stage part can be used to test any semantic road segmentation model build on pytorch (nn.Module).

## Summary of ReadMe

1. Structure of the project
2. Test with anacoda
3. Test with Jupyternotebook 
4. Others


## 1. Structure of project
  ```
RoadSegmentation/
│
├── ReadMe.md             - introduces RoadSegmentator
├── RoadSegmentation.yml  - library dependence
├── roadSeg_Demo.ipynb    - Jupyternotebook file to show how to realize road segmentation with RoadSegmentator
├── downloadMASS.py       - script to download Massachusetts dataset
│
├── source/               - folder of codes
│     ├── utils.py        - contains general functions that are required from other code files
│     ├── networks.py     - contains all the nerual networks used in this project (users can define their own neural networks with pytorch here)
│     ├── genSamples.py   - contains functions to generate training samples (e.g. include different augmentation functions)
│     ├── segmentator.py  - defines the segmentator class, which manages the training and testing processes               
│     └── testDemo.py     - shows how to run the complete two-stage segmentation method  
│
├── tools/                - folder of functions to download MASS dataset online
│     └── mulLinkDown.py  - contains functions to download MASS dataset online
│
├── models/               - the folder to save all the trained models
│     ├── xxx.pth         - file of trained models 
│     └── trainlog/       - the folder to saves all the training history
│           └── xxx_histlog.npy - saved training history file
│
├── data/                 - the folder to save dataset, user need to download or put their dataset in this folder
│     └── MASS/           - folder of an example of dataset (here is MASS dataset)
│           ├── TrainX/   - folder of training X images
│           │     └──  <train1>.tiff - training X image Note: the postfix and image file type can be different, see testDemo.py
│           ├── TrainY/   - folder of training Y images
│           │     └──  <train1>.tif  - training Y image
│           ├── TestX/    - folder of test X images
│           │     └──  <test1>.tiff  - test X image
│           └── TestY/    - folder of test Y images
│                 └──  <test1>.tif   - test Y image
│
└── output/              - the folder to save test segmentation results
  ```
## 2. Test with anacoda

To test the code with Anaconda:

**Step1.** User needs to install [Anaconda](https://www.anaconda.com) first

**Step2.** Then creating an virtual environment with required packages by the command:

&nbsp;&nbsp;&nbsp;&nbsp;   ***conda env create -f RoadSegmentation.yml***

     Note: the versions of the dependent libraries are the ones used in our test. User may need to change the version according to your machine.
    
**Step3.** Activate the environment:

&nbsp;&nbsp;&nbsp;&nbsp;   ***source activate RoadSegmentation***
    
**Step4.** Download the test dataset from [Massachusetts](https://www.cs.toronto.edu/~vmnih/data/) by running:

Install two packages mechanize and requests first:

mechanize: ***conda install -c conda-forge mechanize***

requests:  ***conda install -c conda-forge requests***

Then run &nbsp;&nbsp;&nbsp;&nbsp;   ***python3.x downloadMASS.py*** 
    
    Note: 3.x is user's python version
    
**Step5.** Test the full 2-Stage approach by running the demo file:

&nbsp;&nbsp;&nbsp;&nbsp;   ***cd ./source/***
   
&nbsp;&nbsp;&nbsp;&nbsp;   ***python3.x testDemo.py***
   
    Note: testDemo.py shows a complete test example (include, generate, training, test in 1st stage and 2nd stage respectively) 

## 3. Test with jupyternotebook

A test demo is also presented in **roadSeg_Demo.ipynb**

To run the junpternotebook file:

**Step1.** &nbsp; ***source activate RoadSegmentation***     

%activate the enviornment

**Step2.** &nbsp; ***conda install -c anaconda ipykernel***

**Step3.** &nbsp; ***python -m ipykernel install --user --name=RoadSegmentation***  

%Load the environment for jupyternotebook

**Step4.** &nbsp; ***jupyter-notebook***

%Run jupyter-notebook in the project directory __/RoadSegmentation/

**Step5.** &nbsp; Open the **roadSeg_Demo.ipynb** file in web browser

%Make sure the kernel is RoadSegmentation (can be set in the Kernel button)




[python-image]: https://img.shields.io/badge/Python-3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/static/v1?label=Pytorch&message=1.12.1&color=<green>
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/github/license/CosmosYu9/RoadSegmentation
[lic-url]: #
