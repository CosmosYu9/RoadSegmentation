#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""************************************************************
*  Augmentation to training images                            *
*  Latest update: Feb 1st, 2021                               *
*  By Cosmos&yu                                               *
************************************************************"""
from scipy import ndimage
import numpy as np
from scipy import signal
import os, os.path
from skimage.io import imread
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import utils as ut
import torch
from skimage.morphology import disk


#Flip Images
#Flip the image right to left/ or uppper and down
def FlipImg(colorImg, GTImg, isRLorUD = 0, dT = 10):
    
    dstCImg = np.flip(colorImg, isRLorUD)
    dstGTImg = np.flip(GTImg, isRLorUD)
    
    dstCImg[dstCImg > 255] = 255
    dstCImg[dstCImg < 0] = 0
    
    dstGTImg[dstGTImg > dT] = 255
    dstGTImg[dstGTImg < 255]   = 0    
    
    return dstCImg, dstGTImg 

#Rotate Image with angele RotateAng
def ClockRotateImg(colorImg, GTImg, rotateAng = 0, bT = 10):
    
    dstCImg = ndimage.rotate(input = colorImg, angle = rotateAng)
    dstGTImg = ndimage.rotate(input = GTImg, angle = rotateAng)
        
    dstCImg[dstCImg > 255] = 255
    dstCImg[dstCImg < 0] = 0
    
    dstGTImg[dstGTImg > bT] = 255
    dstGTImg[dstGTImg < 255]   = 0
    
    return dstCImg, dstGTImg 

#DarkingImg
def DarkingImg(colorImg, GTImg, darkingRatio = 1.0, bT = 10):
    
    dstCImg = np.copy(colorImg)
    dstGTImg = np.copy(GTImg)
    
 
    dstCImg[:, :, : ] = np.uint(colorImg[:, :, :] * darkingRatio)

    dstCImg[dstCImg > 255] = 255
    dstCImg[dstCImg < 0] = 0
    
    dstGTImg[dstGTImg > bT] = 255
    dstGTImg[dstGTImg < 255]   = 0

    return dstCImg,dstGTImg 

#Randomly select blocks with size sizeS*sizeS from training images
def RandomSelect256Blocks(orgImg, GTImg, sampleX, sampleY, countS = [0], number = 20, sizeS = 256, isContEmpty = True):

    width = GTImg.shape[1]
    height = GTImg.shape[0]
    
    lw = width - sizeS - 10
    lh = height - sizeS - 10
    
    hsizeS = int(sizeS / 2)
    
    for i in range(number): 
        
         cenX =  int(np.random.uniform(0, lw) + hsizeS)
         cenY =  int(np.random.uniform(0, lh) + hsizeS)
         
         sampleX[countS[0]] = orgImg[cenY - hsizeS : cenY + hsizeS, cenX - hsizeS : cenX + hsizeS,:]
         sampleY[countS[0], :, :, 0] = GTImg[cenY - hsizeS : cenY + hsizeS, cenX - hsizeS : cenX + hsizeS]
         
         if(isContEmpty == True):
             if(np.sum(GTImg[cenY - hsizeS : cenY + hsizeS, cenX - hsizeS : cenX + hsizeS]) == 0):
                 continue
         
         countS[0] += 1


#Generate training samples for segmentation
def genTrainS4Seg(orgImgDir='../data/MassSel/SelectedTrainX/', 
                  GTDir = '../data/MassSel/SelectedTrainY/', 
                  nameSpliChar = '_',
                  saveDir = '../data/MassSel/TrainSamples/', 
                  gtPosix = '.tif', maxNum = 1000, K = [30, 20, 20, 20, 20], rotate = 1, sampleSize = 256,
                  saveAsMat = True):
    
     fileList = os.listdir(orgImgDir) #Get the file names of the training images
     
     trainOrgNum = min(maxNum, len(fileList))
     
     storeGB = (trainOrgNum * (np.sum(K[: -1]) + rotate * K[-1]) * sampleSize * sampleSize * 1 * 4)/ (1024 ** 3)
     
     print('The estimated memory needed is:', storeGB)
     
     totalSamples = trainOrgNum * (np.sum(K[: -1]) + rotate * K[-1])
     
     print(totalSamples)
     
     sampleX = np.zeros((totalSamples, sampleSize, sampleSize, 3), dtype = 'uint8')
     sampleY = np.zeros((totalSamples, sampleSize, sampleSize, 1), dtype = 'uint8')
     
     countS = [0]
     
     print('Total # of ORG images is:', trainOrgNum, '[', end = '', flush = True)
   
    
     if(os.path.isdir(saveDir) == False):
            os.makedirs(saveDir)  

     start_0 = time.time() 
     for i in range(trainOrgNum):
                  
         start_t = time.time()  
         
         orgfile = fileList[i]
         prefile = orgfile.split(nameSpliChar)[0]
         gtfile = prefile + gtPosix
         
         orgfile = orgImgDir + orgfile
         gtfile = GTDir + gtfile   #Get the ground truth image
         
         if(os.path.exists(gtfile) == False):
             continue

         print('Current process on: ', prefile, i, '/', trainOrgNum )
         GTImg = imread(gtfile)
         orgImg = imread(orgfile)
         
         
         if(len(GTImg.shape) > 2):
             GTImg  = ut.rgb2gray(GTImg)
             GTImg[GTImg > 128] = 255
             GTImg[GTImg < 255] = 0
         
#         print(GTImg.dtype, np.max(GTImg), GTImg.shape)
#         print(orgImg.dtype,np.max(orgImg),orgImg.shape)
                           
         RandomSelect256Blocks(orgImg, GTImg, sampleX, sampleY, countS = countS, number = K[0], sizeS = sampleSize, isContEmpty = False)
             
##Variation on Flip
         dstCImg, dstGTImg = FlipImg(orgImg, GTImg, 0)
         RandomSelect256Blocks(dstCImg, dstGTImg, sampleX, sampleY, countS = countS, number = K[1], sizeS = sampleSize)
 
         dstCImg, dstGTImg = FlipImg(orgImg, GTImg, 1)
         RandomSelect256Blocks(dstCImg, dstGTImg, sampleX, sampleY, countS = countS, number = K[2], sizeS = sampleSize)
         
#Darkling                 
         dstCImg, dstGTImg = DarkingImg(orgImg, GTImg,  0.1)#np.random.uniform(0.4, 1.2))
         RandomSelect256Blocks(dstCImg, dstGTImg, sampleX, sampleY, countS = countS, number = K[3], sizeS = sampleSize)

#Rotate:
         for  j in range(rotate):
             
             dstCImg, dstGTImg = ClockRotateImg(orgImg, GTImg, rotateAng = np.random.uniform(1, 359))
             RandomSelect256Blocks(dstCImg, dstGTImg, sampleX, sampleY, countS = countS, number = K[4], sizeS = sampleSize)
            
         end_t=time.time() 
         print('Current image processing time is:' ,end_t - start_t )
         print('Acc processing time is:', end_t - start_0)         
    

     print('] , Complete')

    
     sampleX = sampleX[ : countS[0]]
     sampleY = sampleY[ : countS[0]]
     print(sampleX.shape, sampleY.shape)
   
     sampleX = np.swapaxes(sampleX, axis1 = 1, axis2 = 3) #Convert the dimension of sampleX to (Num, Dim, Height, Width)
     sampleY = np.swapaxes(sampleY, axis1 = 1, axis2 = 3)    
     
     if(saveAsMat == True):
         
         sampleX, sampleY = ut.RandomShuffle(sampleX, sampleY, yDim = 1)
    
         print('SampleX shape-type:', sampleX.shape, sampleX.dtype)
         print('SampleY shape-type:', sampleY.shape, sampleY.dtype)
         
         np.save(saveDir + 'SampleX.npy', sampleX)
         np.save(saveDir + 'SampleY.npy', sampleY)
    
     else:
         for n in range(sampleX.shape[0]):
             np.save(saveDir + 'sample' + str(n) + '_sat.npy',  sampleX[n])   
             np.save(saveDir + 'sample' + str(n) + '_mask.npy', sampleY[n])     
             
     return
 


def LoadTrainSamples(sampleDir = '../data/MassSel/TrainSamples/', XName = 'SampleX', YName = 'SampleY'): #The train X,Y samples is save in the same directory with different names
    
    sampleX = np.load(sampleDir + XName + '.npy')
    sampleY = np.load(sampleDir + YName + '.npy')
        
#    print('sampleX dimension:', sampleX.shape, 'Type:', sampleX.dtype, 'Max:', np.max(sampleX))
#    print('sampleY dimension:', sampleY.shape, 'Type:', sampleY.dtype, 'Max:', np.max(sampleY))
    
    return sampleX, sampleY


def ViewTrainSamples(sampleDir = '../data/MassSel/TrainSamples/', viewDir = '../data/MassSel/ViewSamples/', XName = 'SampleX', YName = 'SampleY', checkNum = 5):
    
    sampleX, sampleY = LoadTrainSamples(sampleDir = sampleDir,  XName = XName, YName = YName)
 
    sampleX = np.swapaxes(sampleX, axis1 = 1, axis2 = 3) #Convert the dimension of sampleX to (Num, Dim, Height, Width)
    sampleY = np.swapaxes(sampleY, axis1 = 1, axis2 = 3)
    
    if(os.path.isdir(viewDir) == False):
            os.makedirs(viewDir)  
    else:
            os.system('rm ' + viewDir + '*')  
    
    num = min(checkNum, sampleY.shape[0])
    
    for n in range(num):
        
         print('save sample:', n)
         cImg = sampleX[n] 
         gtImg = sampleY[n, : , : , 0]
         
         if(sampleY.shape[3] > 1): #Save the visulaized weight map
             weightImg = sampleY[n, : , : , 1]
             plt.imsave(viewDir + str(n) + '_SampleW.png', weightImg)             
             
         dstImg = np.copy(cImg)
         dstImg[:, :, 2] = dstImg[:, :, 2] + gtImg * 0.2
         dstImg[dstImg > 255] = 255
         plt.imsave(viewDir + str(n) + '_SampleZ.png', dstImg)    
         plt.imsave(viewDir + str(n) + '_SampleX.png', cImg)
         
         gt3Img = np.zeros((gtImg.shape[0], gtImg.shape[1], 3), dtype = 'uint8' )
         gt3Img[: , :, 0] = gtImg; gt3Img[: ,:, 1] = gtImg; gt3Img[: ,:, 2] = gtImg
         plt.imsave(viewDir + str(n) + '_SampleY.png', gt3Img)
         
'''******The code block below realized the Richer UNet's distance/weight MAP for loss function*************  '''         

#Add distant map to the training Y files, just for training
#Realize the edge based loss function defined in Richer UNet by weight map
#Step 1. Calculate the edge of ground truth Y first
#Step 2. Calculate the distance map for Y
#Step 3. Calculate the weight map for Y
  
def AddDistMap2Y(sampleDir = '../data/MassSel/TrainSamples/', XName = 'SampleX', YName = 'SampleY', SaveName = 'SampleYD',
                 T = 0.3, alpha = 3, p = 5):
    
    sampleX, sampleY = LoadTrainSamples(sampleDir = sampleDir, XName = XName, YName = YName)
    
    numSamples = sampleY.shape[0]

    sampleYD = []
    
    print('Total numSamples is:', numSamples)
    
    for n in range(numSamples):
        
        print(n, '/', numSamples)
        
        Y = sampleY[n, 0]
        
        edgeY = ExtractEdge(Y, T = T)
        
        weightY = ExpandDistMap(edgeY, alpha = alpha, p = p)
    
        Y = np.expand_dims(Y, axis = 0)
        weightY = np.expand_dims(weightY, axis = 0)
        
        sampleYD.append(np.concatenate((Y, weightY), axis = 0))
        sampleYD = sampleYD.astype('float32')
    sampleYD = np.array(sampleYD)
    
    print('Size of sampleYD:', sampleYD.shape)
    
    np.save(sampleDir + SaveName + '.npy', sampleYD)
#Detect edge pixles
def ExtractEdge(Y, T = 0.3):

    
    gradFilter = np.array([[ -1 - 1j, 0 - 2j,  +1 - 1j],
                           [ -2 + 0j, 0 + 0j,  +2 + 0j],
                           [ -1 + 1j, 0 + 2j,  +1 + 1j]]) # Gx + j*Gy   
    
    gradY = signal.convolve2d(Y, gradFilter, boundary = 'symm', mode = 'same') 
    #Note: use symmetric boundary condition to avoid creating edges at the image boundaries.
        
    edgeY = np.abs(gradY)  
    
    edgeY[edgeY > T] = 1
    edgeY[edgeY < 1] = 0
    
    return edgeY
         
def ExpandDistMap(edgeY, alpha = 3, p = 5): #edgeY is a binary matrix

#Step 1. Caculate the distance map by BSF
    rows = edgeY.shape[0]
    cols = edgeY.shape[1]
    
    distMap = np.copy(edgeY)
    
    queBSF = []
    
    #Put the seeds into que first
    for r in range(rows):
        for c in range(cols):
            
           if(edgeY[r, c] == 1):
               
               queBSF. append((r, c))
                
    while(len(queBSF) > 0):
        
        r_tmp = queBSF[0][0]
        c_tmp = queBSF[0][1]

        val_temp = distMap[r_tmp, c_tmp]
        #Check its 4 neigbors
    
        r_new = r_tmp - 1; c_new = c_tmp                 
        if(r_new < rows and r_new >= 0 and c_new < cols and c_new >= 0 and distMap[r_new, c_new] == 0): 

            if((val_temp + 1) <= p ):
                queBSF.append((r_new, c_new))
                distMap[r_new, c_new] = val_temp + 1

            else:
                break
            
        r_new = r_tmp + 1; c_new = c_tmp            
        if(r_new < rows and r_new >= 0 and c_new < cols and c_new >= 0 and distMap[r_new, c_new] == 0): 
            
            if((val_temp + 1) <= p ):
                queBSF.append((r_new, c_new))
                distMap[r_new, c_new] = val_temp + 1

            else:
                break
#    
        r_new = r_tmp; c_new = c_tmp - 1                
        if(r_new < rows and r_new >= 0 and c_new < cols and c_new >= 0 and distMap[r_new, c_new] == 0): 
            
            if((val_temp + 1) <= p ):
                queBSF.append((r_new, c_new))
                distMap[r_new, c_new] = val_temp + 1
            else:
                break
            
        r_new = r_tmp; c_new = c_tmp + 1           
        if(r_new < rows and r_new >= 0 and c_new < cols and c_new >= 0 and distMap[r_new, c_new] == 0): 
            
            if((val_temp + 1) <= p ):
                queBSF.append((r_new, c_new))
                distMap[r_new, c_new] = val_temp + 1
            else:
                break
        #Pop the first element
        
        queBSF.pop(0)
        

#Step 2. Calculated the weight map 1+g(d_i) from paper "Richer-UNet"     
        
    distMap[distMap == 0] = 50000
    distMap -= 1
    

    
    distMap = alpha * np.exp( -distMap / p)
    distMap[distMap < 0.001] = 0
      
    distMap += 1 
    
    return distMap

#Test Edge Detection
def TestEdgeDet(imgFile = '../output/SegResult/11128870_15_pred.png'):

    orgImg = imread(imgFile)
    Y = np.dot(orgImg[..., : 3], [0.299, 0.587, 0.114]) / 255.0
    
    Y = Y.astype('float32')
    
    plt.figure(figsize=(12, 12), dpi = 80)
    
    edgeY = ExtractEdge(Y, T = 0.3)
    
    distMap = ExpandDistMap(edgeY, alpha = 3, p = 5)
    
    plt.imshow(distMap)

    print(distMap.dtype)
    return distMap

#wMap = TestEdgeDet(imgFile = '../output/SegResult/11128870_15_pred.png')
'''****************************************************************************************************'''  


'''*********The block below is used to generate training sample for 2nd stage inference*************'''


def CreateRawSampleMatrixFromFiles(sampleList, orgDir, gtDir, org_pos = '_sat.npy', gt_pos = '_mask.npy'):
    
    sampleX = []
    sampleY = []
    time_s = time.time()
    for id in sampleList:
        
        X  = np.load(orgDir + id + org_pos) 
        Y  = np.load(gtDir + id + gt_pos)
        
        sampleX.append(X)
        sampleY.append(Y)
    
    sampleX = np.array(sampleX)
    sampleY = np.array(sampleY)
    time_e = time.time()
    
    print('SampleX shape is:', sampleX.shape, 'SampleY shape is:', sampleY.shape)
    print('Reading time is:', time_e - time_s)
    
    return sampleX, sampleY
    
    

#segNet = seg.segmentor(network = 'unet', width = 200, height = 200, inputDim = 3,  
#                       par = seg.Parameters(unet_nfilters = 32, unet_dropout = 0.08, unet_layerNum = 6),
#                       epochs = 300, batch_size = 16, learn_rate = 0.0001, lossSel = 'DBCE')
def genRaw2ndStageTrainSamples(segNet, sampleX, sampleY, saveDir, 
                               XRName = 'SampleX_Raw2', YRName = 'SampleY_Raw2',
                               netDir = '../models/', sampleS = 512, div = 20, testNum = 100000, 
                               wM = 0.99, wFN = 1.0, wElse = 1, wTP = 0.98, T_IoU = 0.25, saveAsMat = True):
        
    segNet.network.eval()
    
    
    if(os.path.isdir(saveDir) == False):
        os.makedirs(saveDir)  
    
    time_s = time.time()
    #[num, 3, H, W] Note: H should equalt to W = sampleS
    numSamples = min(testNum, sampleX.shape[0])
        
    shapeX = sampleX.shape
    
    sampleXR = np.zeros((shapeX[0], shapeX[1] + 1, shapeX[2], shapeX[3]), dtype = 'uint8')
    sampleYR = np.zeros((shapeX[0], 2,             shapeX[2], shapeX[3]), dtype = 'uint8')
    
    print('Total process samples:', numSamples)
    print('Begin generating: [', end = '-', flush = True)
    
    ndiv = int(numSamples / div)
    count = 0 
    tIoU = 0 
    
    for n in range(numSamples):
        
        if(n != 0 and ndiv != 0 and n % ndiv == 0):
            print('-', end = '', flush = True)
        
        testX = sampleX[n: n + 1]
        
        tensor_x = torch.Tensor(testX / 255.0).to(segNet.device)
        
        net_out = segNet.network(tensor_x)
        
        predY = net_out.cpu().detach().numpy().reshape(sampleS, sampleS)
        testY = sampleY[n, 0]
                 
        testY = ut.BinarizedImg(testY, T = 128)
        binY = ut.BinarizedImg((predY * 255.0).astype('uint8'), T = 128)
        #Set the weight map
        diffY = testY * 1.0 - binY 
 

        WY = np.zeros((sampleS, sampleS), dtype = 'uint8')
        
        
        WY[:] = wElse * 255
        WY[predY  > 0.05] =  wM * 255
#        WY[diffY > 0.8] = wFN * 255       
#        WY[:] = wElse * 255
#        WY[testY.astype(bool) & binY.astype(bool)] = wTP * 255  
               
        diffY = WY.astype('uint8')
###########################3        
        
        IoU = ut.CaculateIoUScore(gtImg = testY, binImg = binY, isP = False);
        prec, recall, F1 = ut.CalculateF1Score(gtImg = testY, binImg = binY)

        
        if(IoU < T_IoU):
            continue

        
        tIoU += IoU
        predY = np.expand_dims(predY, axis = 0); predY = np.expand_dims(predY, axis = 0)
        sampleXR[count : count + 1,    : shapeX[1], :, : ] = testX
        sampleXR[count : count + 1, -1 :          , :, : ] = np.uint(255 * predY)  # Try binary
        sampleYR[count : count + 1, 0] = sampleY[n, 0]
        sampleYR[count : count + 1, 1] = diffY
        
        np.save(saveDir + 'sample' + str(n) + '_sat.npy',  sampleXR[n])   
        np.save(saveDir + 'sample' + str(n) + '_mask.npy', sampleYR[n])         
        count += 1
        
    print('] Finish generating with total time', round(time.time() - time_s, 3))
    
    print('Avg IoU =', tIoU / count)
    
    sampleXR = sampleXR[: count]
    sampleYR = sampleYR[: count]
    
    
    if(saveAsMat == True):
                  
         np.save(saveDir + XRName + '.npy', sampleXR)
         np.save(saveDir + YRName + '.npy', sampleYR)   
    
    else:
        
         for n in range(sampleXR.shape[0]):
             np.save(saveDir + 'sample' + str(n) + '_sat.npy',  sampleXR[n])   
             np.save(saveDir + 'sample' + str(n) + '_mask.npy', sampleYR[n])        
       
    print('sampleX shape and dtype:',  sampleX.shape,  sampleX.dtype)
    print('sampleY shape and dtype:',  sampleY.shape,  sampleY.dtype)
    print('sampleXR shape and dtype:', sampleXR.shape, sampleXR.dtype)    
    print('sampleYR shape and dtype:', sampleYR.shape, sampleYR.dtype)  

#segNet = seg.segmentor(network = 'unet', width = 200, height = 200, inputDim = 3,  
#                       par = seg.Parameters(unet_nfilters = 32, unet_dropout = 0.08, unet_layerNum = 6),
#                       epochs = 300, batch_size = 16, learn_rate = 0.0001, lossSel = 'DBCE')
def genRaw2ndStageTrainSamplesD(segNet, sampleDir,  orgDir, gtDir, saveDir, 
                                org_pos = '_sat.npy', gt_pos = '_mask.npy',
                                XRName = 'SampleX_Raw2', YRName = 'SampleY_Raw2',
                                netDir = '../models/', sampleS = 512, div = 20, testNum = 100000, 
                                wM = 0.99, wElse = 1, delta = 0.05, T_IoU = 0.25, saveAsMat = True):
        
    
    allList = os.listdir(sampleDir)
    sampleList = [f.split('_')[0] for f in allList if f.find('sat') != -1]  
    
    segNet.network.eval()
    
    
    if(os.path.isdir(saveDir) == False):
        os.makedirs(saveDir) 
    else:
        os.system('rm ' + saveDir + '*')
    
    time_s = time.time()
    #[num, 3, H, W] Note: H should equalt to W = sampleS
    numSamples = min(testNum, len(sampleList))
        
    
    print('Total process samples:', numSamples)
    print('Begin generating: [', end = '-', flush = True)
    
    ndiv = int(numSamples / div)
    count = 0 
    tIoU = 0 
    
    sampleXRM = []
    sampleYRM = []
  
    
    for n in range(numSamples):
        
        if(n != 0 and ndiv != 0 and n % ndiv == 0):
            print('-', end = '', flush = True)
        
        testX = np.load(orgDir + sampleList[n] + org_pos) 
        testY = np.load(gtDir + sampleList[n] + gt_pos)   
        
        testX = np.expand_dims(testX, axis=0)
        testY = np.expand_dims(testY, axis=0)
        
        tensor_x = torch.Tensor(testX / 255.0).to(segNet.device)
        
        net_out = segNet.network(tensor_x)
        
        predY = net_out.cpu().detach().numpy().reshape(sampleS, sampleS)
                 
        testY = ut.BinarizedImg(testY, T = 128)
        binY = ut.BinarizedImg((predY * 255.0).astype('uint8'), T = 128)
        #Set the weight map
        diffY = np.abs(testY * 1.0 - binY)
        
        WY = np.zeros((sampleS, sampleS), dtype = 'uint8')
        diffY = diffY.reshape((sampleS, sampleS))
        WY[:] = wElse * 255
        WY[predY  > delta] =  wM * 255
#        WY[diffY > delta] = wM * 255       
#        WY[:] = wElse * 255
#        WY[testY.astype(bool) & binY.astype(bool)] = wTP * 255  
               
        diffY = WY.astype('uint8')
###########################3        
        
        IoU = ut.CaculateIoUScore(gtImg = testY, binImg = binY, isP = False);
        prec, recall, F1 = ut.CalculateF1Score(gtImg = testY, binImg = binY)

        if(IoU < T_IoU):
            continue

        tIoU += IoU
        predY = np.expand_dims(predY, axis = 0); predY = np.expand_dims(predY, axis = 0)
        
        shapeX = testX.shape
        
        sampleXR = np.zeros((shapeX[1] + 1, shapeX[2], shapeX[3]), dtype = 'uint8')
        sampleYR = np.zeros((2,             shapeX[2], shapeX[3]), dtype = 'uint8')
        
        sampleXR[ : shapeX[1], :, : ] = testX[0]
        sampleXR[ -1 :          , :, : ] = np.uint(255 * predY)  # Try binary
        sampleYR[0] = np.uint(testY[0] * 255) 
        sampleYR[1] = diffY
        
        if(saveAsMat):
            
            sampleXRM.append(sampleXR)
            sampleYRM.append(sampleYR)
            
        else:
      
            np.save(saveDir + 'sample' + str(count) + '_sat.npy',  sampleXR) 
            np.save(saveDir + 'sample' + str(count) + '_mask.npy', sampleYR)   
        

        count += 1
        
    print('] Finish generating with total time', round(time.time() - time_s, 3))
    
    print('Avg IoU =', tIoU / count)
    print('Total generated train number is: ', count)
    if(saveAsMat == True):
                  
         sampleXRM = np.array(sampleXRM)
         sampleYRM = np.array(sampleYRM)

         print('sampleXR shape and dtype:', sampleXRM.shape, sampleXRM.dtype)    
         print('sampleYR shape and dtype:', sampleYRM.shape, sampleYRM.dtype) 
         
         np.save(saveDir + XRName + '.npy', sampleXRM)
         np.save(saveDir + YRName + '.npy', sampleYRM)   
    
      
      
        
   

def ChangeSampleWeights(sampleDir = '../data/MassSel/TrainSamples512/', YRName = 'SampleY_Raw2', XRName = 'SampleX_Raw2',
                        wM = 0.99, wFN = 1.0, wElse = 0.98, div = 10):

    time_s = time.time()
    #Load the training samples for the 1st stage network
    sampleXR, sampleYR = LoadTrainSamples(sampleDir = sampleDir, XName = XRName, YName = YRName)   
    
    sampleOrgY = np.copy(sampleYR)
    print('Change sampleX:', sampleXR.shape, 'sampleY:', sampleYR.shape)  
    
    numSamples = sampleXR.shape[0]

    print('[-', end = '', flush = True)
    ndiv = int(numSamples / div)
        
    for n in range(numSamples):
  
        if(n != 0 and ndiv != 0 and n % ndiv == 0):
            print('-', end = '', flush = True)
            
        Y  = sampleYR[n, 0]
        WM = sampleXR[n, -1]
        WY = sampleYR[n, 1]
        
        WY[:] = wElse * 255
        WY[WM > 200] =  wM * 255
        WY[Y * 1.0 - WM > 128] = wFN * 255
    
    print(np.sum(sampleYR), np.sum(sampleOrgY))
    print('sampleYR shape and dtype:', sampleYR.shape, sampleYR.dtype)  
    print('sampleYR shape and dtype:', sampleYR.shape, sampleYR.dtype)     
    
    np.save(sampleDir + YRName + '.npy', sampleYR) 
    time_e = time.time()
    
    print('Process time is:', time_e - time_s )
        
        
    

def CheckSampleQuality4_2ndStage(sampleDir = '../data/MASS/TrainSamples_SC2/', viewCheckDir = '../data/MASS/ViewCheck/',
                                 YName = 'SampleY_Raw2', XRName = 'SampleX_Raw2', checkNum = 100):
    
    time_s = time.time()
    #Load the training samples for the 1st stage network
    sampleXR, sampleY = LoadTrainSamples(sampleDir = sampleDir, XName = XRName, YName = YName)  
    
    print('MinMax of WY:', np.min(sampleY[:, 1,:, :]), np.max(sampleY[:, 1,:, :]))
    
    print('Check sampleX:', sampleXR.shape, 'sampleY:', sampleY.shape)
    
    sampleXR = np.swapaxes(sampleXR, axis1 = 1, axis2 = 3)
    sampleY  = np.swapaxes(sampleY, axis1 = 1, axis2 = 3)    
    
    numSamples = sampleXR.shape[0]
    checkNum = min(numSamples ,checkNum)
    
    orderList = np.arange(numSamples)
#    np.random.shuffle(orderList) #Randomly check first checkNum samples
    
    if(os.path.isdir(viewCheckDir) == False):
        os.makedirs(viewCheckDir) 
    else:
        os.system('rm ' + viewCheckDir + '*')  
    
    count = 0 
    totalIoU = 0
    for n in orderList[ : checkNum]:
        
        print(count, '/', checkNum)
        
        cImg =  sampleXR[n, :, :, : 3]
        
        maskY = sampleXR[n, :, :, -1]
        gtY   =  sampleY[n, :, :, 0]
        weightY = sampleY[n, :, :, -1]

        dstImg = np.copy(cImg)
        dstImg[:, :, 2] = dstImg[:, :, 2] + maskY * 0.2
        dstImg[dstImg > 255] = 255
        
        gtImg = gtY
        gt3Img = np.zeros((gtImg.shape[0], gtImg.shape[1], 3), dtype = 'uint8' )
        gt3Img[: , :, 0] = gtImg; gt3Img[: ,:, 1] = gtImg; gt3Img[: ,:, 2] = gtImg
        
        wImg = weightY
        wImg[wImg < 250] = 0
        w3Img = np.zeros((wImg.shape[0], wImg.shape[1], 3), dtype = 'uint8' )
        w3Img[: , :, 0] = wImg; w3Img[: ,:, 1] = wImg; w3Img[: ,:, 2] = wImg
        
        mImg = maskY
        m3Img = np.zeros((mImg.shape[0], mImg.shape[1], 3), dtype = 'uint8' )
        m3Img[: , :, 0] = mImg; m3Img[: ,:, 1] = mImg; m3Img[: ,:, 2] = mImg
        
        plt.imsave(viewCheckDir + str(count) + '_SampleZ.png', dstImg)    
        plt.imsave(viewCheckDir + str(count) + '_SampleX.png', cImg)
        plt.imsave(viewCheckDir + str(count) + '_SampleY.png', gt3Img) 
        plt.imsave(viewCheckDir + str(count) + '_SampleM.png', m3Img)
        plt.imsave(viewCheckDir + str(count) + '_SampleW.png', w3Img)

        count += 1
        
        maskY = ut.BinarizedImg(maskY, T = 128)
        gtY = ut.BinarizedImg(gtY, T = 128)
        
        totalIoU += ut.CaculateIoUScore(gtY, maskY, r = 4)   
        ut.CalculateF1Score(gtY, maskY, r = 4)   
        
        
    print('total process time is:',  round(time.time() - time_s), 3)
    print('Avg IoU of random ', checkNum, 'of samples are: ', totalIoU / checkNum )
        
        
        

#Random Hole algorithm, generate random holes on the road pixels
#XSample is the X training samples we created, it has shape:
#It is just one sample, Height must equal to width for our task
'''(1, Dimension = 4 (RGB+stage1 pred), Height, Width)'''
#YSample is the Y training samples, it has shape:
'''(1, Dimension = 2(GT + Weights),     Height, Width)'''

#Our goal is to generate n holes on the last dimension of XSample

def RandomHoles(XSample, YSample, N, hr_min = 50, hr_max = 100):
    
    
     sampleS = XSample.shape[2]
     
     gtY = YSample[0, 0]
     
     gtPix = np.where(gtY >= 220) 
        
     gtPts = [ [gtPix[0][i], gtPix[1][i]]  for i in range(len(gtPix[0]))]

     
     rdIndex = np.arange(len(gtPix[0]))
     np.random.shuffle(rdIndex)
     
     nTypes = np.random.randint(0, 1, N) #Generate the types of holes; currently, we have two types of hole: 0 - disk; 1 - square 
     nRadius = np.random.randint(hr_min, hr_max + 1, N)
     
     for n in range(N):
         
         n_id = gtPts[rdIndex[n]] #Get the id of hole

#Get the center of hole
         n_x = n_id[0]
         n_y = n_id[1]
         n_t = nTypes[n]  #Get the random type
         n_r = nRadius[n] #Get the random radius 
                  
#If the hole is outside the image then continue
         
         if(n_x - n_r < 0 or n_y - n_r < 0 or n_x + n_r >= sampleS or n_y + n_r >= sampleS):
             continue
         
         patchX = XSample[0, 3, n_x - n_r : n_x + n_r + 1, n_y - n_r : n_y + n_r + 1]
         patchRGB = XSample[0, : 2, n_x - n_r : n_x + n_r + 1, n_y - n_r : n_y + n_r + 1] #Change the rgb according to y
         patchY = YSample[0, 0, n_x - n_r : n_x + n_r + 1, n_y - n_r : n_y + n_r + 1]
         
         if(n_t == 1):
             patchX[:] = 0  #Just set all data in the square to be 0s
         else:
             patchX[:] = patchX *  (1 - disk(n_r))

        

def AddRandomHoles(sampleDir = '../data/MASS/TrainSamples_SC2/', YRName = 'SampleY_Raw2', XRName = 'SampleX_Raw2', 
                        NXRName = 'SampleX_Raw2N',
                        NRange = 6, hr_min = 8, hr_max = 50,
                        div = 10, minN = 100000):

    time_s = time.time()
    #Load the training samples for the 1st stage network
    sampleXR, sampleYR = LoadTrainSamples(sampleDir = sampleDir, XName = XRName, YName = YRName)   

    print('Change sampleX:', sampleXR.shape, 'sampleY:', sampleYR.shape)  
    
    numSamples = min(minN, sampleXR.shape[0])

    print('[-', end = '', flush = True)
    ndiv = int(numSamples / div)
        
    for n in range(numSamples):
  
        if(n != 0 and ndiv != 0 and n % ndiv == 0):
            print('-', end = '', flush = True)
            
        
        N = np.random.randint(0, NRange)
        
        RandomHoles(sampleXR[n : n + 1], sampleYR[n : n + 1], N, hr_min = hr_min, hr_max = hr_max)
    

    print('sampleYR shape and dtype:', sampleYR.shape, sampleYR.dtype)  
    print('sampleYR shape and dtype:', sampleYR.shape, sampleYR.dtype)     
    
    np.save(sampleDir + NXRName + '.npy', sampleXR)
    
    time_e = time.time()
    
    print('Process time is:', time_e - time_s )


def Merge2TrainSamples(fileX1 = '../data/MASS/TrainSamples_SC/SampleX_Raw1.npy', 
                       fileY1 = '../data/MASS/TrainSamples_SC/SampleY_Raw1.npy',
                       fileX2 = '../data/MASS/TrainSamples_SC2/SampleX_Raw2.npy', 
                       fileY2 = '../data/MASS/TrainSamples_SC2/SampleY_Raw2.npy', 
                       saveX  = '../data/MASS/TrainSamples_SC2/SampleX_RawM.npy', 
                       saveY  = '../data/MASS/TrainSamples_SC2/SampleY_RawM.npy'):
    
    sampleX1 = np.load(fileX1)
    
    sampleY1 = np.load(fileY1)
    
    sampleX2 = np.load(fileX2)
    
    sampleY2 = np.load(fileY2)
    
    print('X1:', sampleX1.shape, 'Y1:', sampleY1.shape)
    print('X2:', sampleX2.shape, 'Y2:', sampleY2.shape)
        
    sampleX = np.concatenate((sampleX1, sampleX2), axis = 0)
    
    sampleY = np.concatenate((sampleY1, sampleY2), axis = 0)
    
    print('MergeX', sampleX.shape, 'MergeY:', sampleY.shape) 

    np.save(saveX, sampleX)
    np.save(saveY, sampleY)
    
def CvtS2TrainSampleToS1(fileX = '../data/MASS/TrainSamples_SC2/SampleX_RawM.npy', 
                         fileY = '../data/MASS/TrainSamples_SC2/SampleY_RawM.npy',
                         saveX  = '../data/MASS/TrainSamples_SC/SampleXM.npy', 
                         saveY  = '../data/MASS/TrainSamples_SC/SampleYM.npy'):

    sampleX = np.load(fileX)
    
    sampleY = np.load(fileY)

    sampleXM = sampleX[:, : 3]

    sampleYM = sampleY[:, : 1]

    print('MergeX', sampleXM.shape, 'MergeY:', sampleYM.shape)  
    
    np.save(saveX, sampleXM)
    np.save(saveY, sampleYM)    
    
#CvtS2TrainSampleToS1()
#Merge2TrainSamples()
    
#AddRandomHoles(sampleDir = '../data/MASS/TrainSamples_SC2/', YRName = 'SampleY_Raw2', XRName = 'SampleX_Raw2', 
#                        NXRName = 'SampleX_Raw2N', div = 10, minN = 20)
#    
#CheckSampleQuality4_2ndStage(sampleDir = '../data/MASS/TrainSamples_SC2/', viewCheckDir = '../data/MASS/ViewCheck/',
#                                 YName = 'SampleY_Raw2', XRName = 'SampleX_Raw2N', checkNum = 20)   

#Test functions:
#genTrainS4Seg(orgImgDir='../data/MassSel/SelectedTrainX/', 
#                   GTDir = '../data/MassSel/SelectedTrainY/', 
#                   nameSpliChar = '.',
#                   saveDir = '../data/MassSel/TrainSamples/', 
#                   gtPosix = '.tif', maxNum = 1000, K = [5, 2, 2 ,2 ,5], rotate = 30, sampleSize = 512)
#
#ViewTrainSamples(sampleDir = '../data/MassSel/TrainSamples/', viewDir = '../data/MassSel/ViewSamples/',
#                 XName = 'SampleX', YName = 'SampleY')


#AddDistMap2Y(sampleDir = '../data/MassSel/TrainSamples/', XName = 'SampleX', YName = 'SampleY', SaveName = 'SampleYD',
#                 T = 0.3, alpha = 3, p = 5)


        
#        setY = sampleXR[n, -1]
#        setY[setY > 0.5] = 1
#        setY[setY < 1] = 0
#        ut.CaculateIoUScore(sampleY[n, 0 ], setY, r = 4)    
#
#        setY = predY[0,0]
#        setY[setY > 0.5] = 1
#        setY[setY < 1] = 0
#        ut.CaculateIoUScore(sampleY[n, 0 ], setY, r = 4) 
