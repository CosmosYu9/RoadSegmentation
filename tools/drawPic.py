# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def DrawK_1DLine(dataK, labels, xyAxis = ['nth data', 'value'], title = ' ', isAnom = False, Anom =[[]], isGT = False, GT = [[]], loc = 'upper right', 
                 colors = ['blue', 'orange', 'green', 'purple', 'red'], figsize = (12, 8), xline = -1, fileName = './123.png', isSave = False):

    title = ' '; #For paper!!!!!!
    matplotlib.rc('font', size = 24) #Set font for saving images
    linestyles = ['-', '--', '-.', ':','--'] #Define line styles 
    
    plt.figure(figsize = figsize) #Set figure size
    plt.xlabel(xyAxis[0])       #Draw x label
    plt.ylabel(xyAxis[1])       #Draw y label
    plt.title(title)            #Draw figure title 


    if(xline != -1):              #Draw a vertical line in place xline 
        plt.axvline(x = xline, linestyle = '--', color = 'k')

    #Draw the K lines 
    for k in range(dataK.shape[1]):
        alpha = 0.9 ** (k+1)
        plt.plot(range(dataK.shape[0]), dataK[:, k], alpha = alpha, label = labels[k], color = colors[k%len(colors)], linestyle = linestyles[k%len(linestyles)])

    #Draw ground truth in shaded area with color green
    
    ymax = np.max(dataK) * 1.3
    ymin = np.min(dataK) - np.abs(np.max(dataK)) * 0.1
    plt.ylim(top = ymax, bottom = ymin)
#    ymin, ymax = plt.ylim()
    end = dataK.shape[0]
#    print(ymin, ymax)

    if(isAnom):
        for i in range(len(Anom)):
            
             plt.axvspan(Anom[i][0], Anom[i][1], alpha = 0.8, color = 'green')  
             
             midline = int((Anom[i][0]+min(Anom[i][1],end))/2)
             widthline = min(Anom[i][1],end) - Anom[i][0]
             plt.bar(midline, width = widthline, height = ymax - ymin, bottom = ymin, hatch='/', color='white', alpha = 1.0, edgecolor='green' )
 
    if(isGT):

        for i in range(len(GT)):
            
#            midline = int((GT[i][0]+min(GT[i][1],end))/2)
#            widthline = min(GT[i][1],end) - GT[i][0]
#            xmin, xmax, ymin, ymax = plt.axis()
#
#            plt.bar(midline,width = widthline, height = ymax - ymin, bottom = ymin, hatch='/', color='white', alpha=0.6, edgecolor='red' )
            plt.axvspan(GT[i][0], GT[i][1], alpha = 0.6, color = 'red')         

    plt.grid(True)
    
    plt.legend(loc = loc)
    if(isSave): #Save figure
           plt.savefig(fileName)
    else:
       plt.show()
    
    plt.close('all')
    

def DrawLoss(lossDir = '../models/trainlog/', modelName = 'MASS_CUNet_RU_DWBCE'):
    
    lossLogFile = lossDir + modelName + '_histlog.npy'
    
    logMat = np.load(lossLogFile)
    
    dataK = logMat[:, : 2]
    DrawK_1DLine(dataK, labels = ['train loss', 'val loss '], xyAxis = ['epoch', 'loss'], title = ' ', isAnom = False, Anom =[[]], isGT = False, GT = [[]], loc = 'upper right', 
                 colors = ['blue', 'orange', 'green', 'purple', 'red'], figsize = (12, 8), xline = -1, fileName = './123.png', isSave = False)

    return dataK


# CU = DrawLoss(lossDir = '../models/trainlog/', modelName = 'MASS_CUNet_RU_DWBCE')
# U32 = DrawLoss(lossDir = '../models/trainlog/', modelName = 'MASS_UNet_RU_DWBCE')
# matplotlib.rc('font', size = 22) #Set font for saving images
# linestyles = ['-', '--', '-.', ':','--'] #Define line styles 
    
# plt.figure(figsize =  (12, 8)) #Set figure size
# plt.xlabel('Number of Epoch')       #Draw x label
# plt.ylabel('DICE + WBCE Loss')       #Draw y label
# plt.plot( CU[:, 0], label = 'CUNet train loss', linestyle = '-')
# plt.plot( U32[:, 0], label = 'UNet32 train loss', linestyle = '--')
# plt.plot( CU[:, 1], label = 'CUNet val loss', linestyle = '-') 
# plt.plot( U32[:, 1], label = 'UNet32 val loss', linestyle = '--') 

# #plt.plot( CU[:, 1], label = 'CUNet', linestyle = '-') 
# #plt.plot( U32[:, 1], label = 'UNet32', linestyle = '--') 
# #plt.axvline(x = np.argmin(CU[:, 1]), linestyle = '-', color = 'red')
# #plt.axvline(x = np.argmin(U32[:, 1]), linestyle = '-', color = 'green')

# plt.legend(loc = 'upper right')
# plt.grid(True)