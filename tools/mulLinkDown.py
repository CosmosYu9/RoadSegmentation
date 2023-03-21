import mechanize
from time import sleep
import requests #The module to download files from website
import os

#Download file from a signle url
def downloadlink(url, saveFile):
    
    r = requests.get(url, allow_redirects = True, stream = True)  #Set stream = True, then can check whether 
                                                                  #the file is downloadable or not by check r.headers ?
    if(r.headers != None):
        open(saveFile, 'wb').write(r.content) #Download the 
        return True
    else:
        return False

def DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/', 
                       filetypes=[".tiff", ".tif"],
                       saveDir = '../data/MASS/TrainY/'):

    #Make a Browser (think of this as chrome or firefox etc)
    br = mechanize.Browser()

    # Open your site
    br.open(site)

    linkfiles=[] #Link objects list

#Filter the link files by filetypes
    for l in br.links(): #you can also iterate through br.forms() to print forms on the page!
        for t in filetypes:
            if t in str(l.url): #check if this link has the file extension we want (you may choose to use reg expressions or something)
                linkfiles.append(l) 
 
    if(os.path.isdir(saveDir) == False):
        os.makedirs(saveDir) 
        
    count = 0
    for l in linkfiles:
        sleep(1)
        fileName = l.text
        fileURL  = l.url
        if(downloadlink(fileURL, saveDir + fileName)):
            print(count, fileName)
            count += 1
    
    print('total files:', count)


# DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat', 
#                       filetypes=[".tiff", ".tif"],
#                       saveDir = '../data/MASS/TestX/')

# DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map', 
#                       filetypes=[".tiff", ".tif"],
#                       saveDir = '../data/MASS/TestY/')

# DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat', 
#                       filetypes=[".tiff", ".tif"],
#                       saveDir = '../data/MASS/TrainX/')

# DownLFilesFromSite(site = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map', 
#                       filetypes=[".tiff", ".tif"],
#                       saveDir = '../data/MASS/TrainY/')