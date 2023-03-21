# -*- coding: utf-8 -*-

"The code script to download the Massachusetts dataset from https://www.cs.toronto.edu/~vmnih/data/mass_roads/"
import tools.mulLinkDown as down
import os


rootDir = './'
dataDir = rootDir + 'data'
if(os.path.isdir(dataDir) == False):
    os.makedirs(dataDir)

massDir = dataDir + '/MASS'
if(os.path.isdir(massDir) == False):
    os.makedirs(massDir)


#Download test X
down.DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat', 
                       filetypes=[".tiff", ".tif"],
                       saveDir = massDir + '/TestX/')
#Download test Y
down.DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map', 
                       filetypes=[".tiff", ".tif"],
                       saveDir = massDir + '/TestY/')
#Download train X
down.DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat', 
                       filetypes=[".tiff", ".tif"],
                       saveDir = massDir + '/TrainX/')
#Download train Y
down.DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map', 
                       filetypes=[".tiff", ".tif"],
                       saveDir = massDir + '/TrainY/')