#Version 2023.3
RoadSegmentator project provides the framework for road segementation in satellite images
It realizes the two stage road segmentation method proposed in [1] T. Li, M. Comer and J. Zerubia, "A Two-Stage Road Segmentation Approach for Remote Sensing Images, " ICPR2022 workshop
For the first stage, user can test different nerual networks on road segmentation problem; 
For the second stage, we can enhance the preliminary segmentation result from stage one by our cunet.

#Summary of File:
    1. Structure of project files;
    2. Installation; 
    3. Test with Jupyternotebook file;
    4. Contact;
    5. Others.

#1. Structure of project files:

RoadSegmentator/
│
├── ReadMe.md             - introduces RoadSegmentator
├── xRoadSegmentator.yml  - library dependence
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
└──  output/              - the folder to save test segmentation results


#2. Installation:
    
    The code can be with Anaconda(https://www.anaconda.com).
    User need to install Anaconda(https://www.anaconda.com) first.
    Then creating an virtual environment with required packages
    the command: 'conda env create -f RoadSegmentator.yml' will create the environment automatically
    Then 'source activate RoadSegmentator'


#The library dependences are listed in ./RoadSegmentator.yml 


#3. Test with Jupyternotebook file
    step1. conda env create -f RoadSegmentator.yml     %This command create the enviornment
    step2. conda install -c anaconda ipykernel
    step3. python -m ipykernel install --user --name=RoadSegmentator       %Load the environment for jupyternotebook
    Step4. source activate RoadSegmentator
    Step5. jupyter-notebook --notebook-dir=_____        % __ should be the directory of where RoadSegmentatorlocated, ____/RoadSegmentator


#4. Contact:
If you have any questions about our work or code, please send Tianyu an email at cosmos.yu9@gmail.com


#Others: Download test dataset
    To download the Massachusetts dataset, need run file ./tools/mulLinkDown.py with python
    Need library mechanize: conda install -c conda-forge mechanize
                 requests: conda install -c conda-forge requests
    Then the MASS data should be saved in file directory: ./data/

#conda install scikit-image
#conda install torchvision