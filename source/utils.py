#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""************************************************************
*  Useful functions                                           *
*  Latest update: Feb 3rd, 2021                               *
*  By Cosmos&yu                                               *
************************************************************"""

"""*********************************************************"""
"""* Functions include:                                     """
"""***********  Group A: Metric Functions        ***********"""                                   
"""* 1. BinarizedImg(predImg, T = 128)                      """
"""* 2. CaculateIoUScore(gtImg, binImg, r = 4)              """
"""* 3. CalculateF1Score(gtImg, binImg, r = 4)              """
"""* 4. IoUFromPreRecall(prec, recall)                      """
"""* 5. F1FromPreRecall(prec, recall)                       """

"""***********  Group B: Load/Save Images data   ***********"""
"""* 6. LoadTestImg(orgImgFile = '.tif', gtImgFile = '.tif', 
                    isLoadGT = False, isOrgImgGray = False) """
"""* 7. LoadGTImg( gtImgFile = '.tif', T = 128)             """
    #DP: Only load ground truth image (binary)
"""* 8. SaveShowImg(predImg, orgImg, predImgName = ' ', 
                    visualImgName = ' ' )                   """
    #DP: Save predicted image
"""* 9. SaveShowSub4Img(predImg, orgImg, saveName = '' )    """
    #DP: Save the predicted image in 4 sub images

"""***********    Group C: Test functions        ***********"""
"""* 10. StatisticsFromSavedResult(groupGTDir,groupPredDir,
                              predType, predPos,r, T)       """
    #DP: Recalculated the quantitative results from saved data
"""* 11. testGroupImages(segNet, groupDir, groupGTDir,
                    gtImgType, saveDir, sampleS, testMax)   """
    #DP: Single stage test segmentation  for images 
    #    under a directory "groupDir"
"""* 12. twoStageTestImages(segNet, segNet2, groupDir, 
                        groupGTDir, gtImgType, saveDir, 
                        sampleS, testMax, T_o )             """
    #DP: Two stage segmentation test

"""* 13. RandomShuffle(trainX, trainY, yDim = 1 )           """
    #DP: random shuffle the training data
"""* 14. RandomShuffle2(train_s, yDim = 1)                  """
"""* 15. SplitTrainImgs(trainXDir, trainYDir, ntrainXDir, 
                   ntrainYDir, ntestXDir, ntestYDir, 
                   nameSpliChar, gtPosix, splitRatio)       """
    #DP: Split training images into two parts according splitRatio
    
"""*********************************************************"""
import numpy as np
from skimage.morphology import binary_opening as opening
import os, os.path
from skimage import exposure,img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.morphology import disk
import matplotlib
import time
matplotlib.use('agg')
import random
import matplotlib.pyplot as plt

'''**************************Metric functions code******************************************'''
#Threshold on the predicted image (output from network) with T
#predImg is a uint8 image, each pixel with value 0-255
def BinarizedImg(predImg, T = 128):
    
    binImg = np.copy(predImg)

    binImg[binImg <= T] = 0
    binImg[binImg > T] = 1
    
    return binImg
    
#Calculate intersection over union score
#r is the print precesion round
def CaculateIoUScore(gtImg, binImg, isP = True, r = 4):  
        
    unImg = gtImg | binImg
    inImg = gtImg & binImg
    
    if(np.count_nonzero(unImg) == 0):
        return 0 
        
    IoU = np.count_nonzero(inImg)/np.count_nonzero(unImg)
    
    if(isP):
        print('IoU = ', round(IoU, r), end = ' ', flush = True)
    
    return IoU
    
#Calculate the TP, FP, FN, prec, recall, F1
#Input gtImg and binImg must be binary images (numpy arrays)
#r is the print precesion round
def CalculateF1Score(gtImg, binImg, isP = False, r = 4):
    
    gtImg = gtImg.astype(int)
    binImg = binImg.astype(int)
        
    TP = np.sum(gtImg & binImg)
    FP = np.sum(binImg - gtImg == 1)
    FN = np.sum(gtImg - binImg == 1)
    if(TP == 0):
        return 0, 0, 0
    prec = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (prec * recall) / (prec + recall)
    
#    print('TP = ', TP, 'FP = ', FP, 'FN = ', FN)
    if(isP):
        print('F1 = ', round(F1 ,r), 'prec = ', round(prec, r), 'recall = ', round(recall, r))
        
    return prec, recall, F1

#Calculate IoU score from prec and recall
def IoUFromPreRecall(prec, recall):
    
    return (prec * recall)/(prec + recall - prec * recall)

#Calculate F1 score from prec and recall
def F1FromPreRecall(prec, recall):
    
    return 2 * (prec * recall) / (prec + recall)

'''**************************************************************************************'''
#Load test image with fileName
def LoadTestImg(orgImgFile = '.tif', gtImgFile = '.tif', isLoadGT = False, isOrgImgGray = False):
    
    output = []
    orgImg = imread(orgImgFile) 
        
    output.append(orgImg)
    
    if(isLoadGT):
        
        GTImg = imread(gtImgFile)

        if(len(GTImg.shape) > 2):
             GTImg  = rgb2gray(GTImg)
             
        GTImg[GTImg < 128] = 0
        GTImg[GTImg > 128] = 255  #Binarized the ground truth image with threshold 0.1
        
        output.append(GTImg)
 
    return output
#Load GT image as  a uint8 numpy matrix, bianry value with 0, 255
def LoadGTImg( gtImgFile = '.tif', T = 128):
    
    GTImg = imread(gtImgFile)
        
    GTImg[GTImg < T] = 0; GTImg[GTImg > T] = 255  #Binarized the ground truth image with threshold 0.1
        
    return GTImg

#Save predicted result (binary), and the orignal image add the predict image in one channel
def SaveShowImg(predImg, orgImg, predImgName = ' ', visualImgName = ' ' ):
             
    dstImg = np.copy(orgImg)
    dstImg[:, :, 2] = dstImg[:, :, 2] + predImg * 0.5
    dstImg[dstImg > 255] = 255
 
    plt.imsave(visualImgName, dstImg)    
    plt.imsave(predImgName, predImg) 
    
#Save predicted result (binary), and the orignal image add the predict image in one channel
def SaveRGB_FPTPFN(orgImg, binImg, gtImg, saveImgName = ' ' ):
             
    
    outImg = np.copy(orgImg)
    gtImg = gtImg.astype(int)
    binImg = binImg.astype(int)
        
    TP_id = (gtImg & binImg == 1)
    FP_id = (binImg - gtImg == 1)
    FN_id = (gtImg - binImg == 1)
    
    rImg = outImg[:,:, 0] 
    gImg = outImg[:,:, 1] 
    bImg = outImg[:,:, 2]
    
    rImg[TP_id] = 0;   gImg[TP_id] = 255; bImg[TP_id] = 0;
    rImg[FP_id] = 255; gImg[FP_id] = 0;   bImg[FP_id] = 0;
    rImg[FN_id] = 0;   gImg[FN_id] = 0;   bImg[FN_id] = 255;
    
    plt.imsave(saveImgName, outImg)    

def SaveShowSub4Img(predImg, orgImg, saveName = '' ):

    dstImg = np.copy(orgImg)
    dstImg[:, :, 2] = dstImg[:, :, 2] + predImg * 0.5
    dstImg[dstImg > 255] = 255

    width = dstImg.shape[0]
    height = dstImg.shape[1]
    
    midW = int(width / 2)
    midH = int(height / 2)

    plt.imsave(saveName + '_UL_vis.png', dstImg[  : midW, : midH])    
    plt.imsave(saveName + '_ULpred.png',   predImg[ : midW, : midH])
    
    plt.imsave(saveName + '_UR_vis.png', dstImg[ midW : , : midH])    
    plt.imsave(saveName + '_UR_pred.png',   predImg[midW : , : midH])

    plt.imsave(saveName + '_DL_vis.png', dstImg[  : midW, midH : ])    
    plt.imsave(saveName + '_DL_pred.png',   predImg[ : midW, midH : ])

    plt.imsave(saveName + '_DR_vis.png', dstImg[ midW :, midH : ])    
    plt.imsave(saveName + '_DR_pred.png',   predImg[ midW :, midH : ])    
    

#Quantified the saved predicted image results (.png)
#r is the print precesion round
def StatisticsFromSavedResult(groupGTDir = '../data/MassSel/TestY/', groupPredDir = '../output/SegResult/',
                              nameSpliChar = '.',
                              predType = '.png', predPos = '_pred2.npy' ,r = 4, T = 128):

    count = 0 #Count how many images are processedx
    IoU_list  = []; F1_list   = [];  prec_list = []; recall_list = [] #List for record the statistic results

    fileList = os.listdir(groupGTDir) #Get the file names of the GT images
    fileNum = len(fileList)
    
    print(fileNum)
    
    for i in range(fileNum):
         
         gtfile = fileList[i]
                 
         prefile = gtfile.split(nameSpliChar)[-2]

         gtfile = groupGTDir + gtfile

#         print('Process image:', prefile)
         
         testY = LoadGTImg(gtImgFile = gtfile)
         
         predYfile = groupPredDir + prefile + predPos
         predY = np.load(predYfile)
         
         binY = BinarizedImg(predY, T = T)
         testY = BinarizedImg(testY, T = T)
         binY = binY.astype('bool')
         testY = testY.astype('bool')
#         IoU = CaculateIoUScore(gtImg = testY, binImg = binY, r = r)
#         IoU = mean_iou(testY, binY)
         prec, recall, F1 = CalculateF1Score(gtImg = testY, binImg = binY, r = r)
#         prec, recall, F1, IoU2 = f1_score(testY, binY)
         
         IoU = IoUFromPreRecall(prec, recall)
#         print('From Pec,Recall IoU=', IoU)
         
         IoU_list.append(IoU);  F1_list.append(F1); prec_list.append(prec);  recall_list.append(recall)
         
         count += 1
         
    meanIoU = np.mean(IoU_list)
    meanF1  = np.mean(F1_list)
    meanPrec = np.mean(prec_list)
    meanRecall = np.mean(recall_list)


    print('Total number of test images is:', count)    
    print('The average statitisc measures are: ')
    print('IoU = ', round(meanIoU, r), ' F1 = ', round(meanF1 ,r), 'prec = ', round(meanPrec, r), 'recall = ', round(meanRecall, r))
    
    return meanIoU, meanF1, meanPrec, meanRecall 

#Test a group of images (eg. from one dataset)
#Note:  1. All test images should be put under the same directory
#       2. All GT images should be put under the same directory
#       3. The TP, FP, FN, Recall, Precision, F1 score and IoU score will be calculated 
#       4. The stitched output from network will be saved in another directory
def testGroupImages(segNet, groupDir = '../data/MASS/TestX/', groupGTDir = '../data/MASS/TestY/',
                    nameSpliChar = '.',
                    gtImgType = '.tif',
                    saveDir  = '../output/SegResult/', sampleS = 512, stepW = -1, testMax = 1000 ):
    

    
    if(stepW == -1):
        
        stepW = int(sampleS / 2)
        
    if(os.path.isdir(saveDir) == False):
            os.makedirs(saveDir)  
            
    fileList = os.listdir(groupDir) #Get the file names of the training images
    fileNum = min(len(fileList), testMax)
    
    count = 0
    IoU_list  = []; F1_list   = [];  prec_list = []; recall_list = [] #List for record the statistic results
    
    for i in range(fileNum):
         
         orgfile = fileList[i]
                 
         prefile = orgfile.split(nameSpliChar)[-2]
         
         gtfile = prefile + gtImgType

         gtfile = groupGTDir  + gtfile   #Get the ground truth image

         orgfile = groupDir + orgfile
         
         if(os.path.exists(gtfile) == False):
             continue
         
         
         print(i, 'Process image:', prefile)
         
         testX, testY = LoadTestImg(orgImgFile = orgfile, gtImgFile = gtfile, isLoadGT = True)
        
         predY = segNet.SegStitch(testX, stepW = stepW, stepH = stepW, unitW = sampleS)

         saveName = saveDir + prefile
         
         SaveShowImg(predY, testX, predImgName = saveName + '_pred.png', 
                        visualImgName = saveName + '_vis.png' )
         
       
#         np.save(saveDir + prefile + '_pred.npy', predY)

         binY = BinarizedImg(predY, T = 128)
         testY = BinarizedImg(testY, T = 128)
         
#         binY = binY.astype('bool')
#         testY = testY.astype('bool')
 
         SaveRGB_FPTPFN(testX, binImg = binY, gtImg = testY, saveImgName = saveName + '_stage1_pred.png' )  
         
         IoU = CaculateIoUScore(gtImg = testY, binImg = binY)
         
         prec, recall, F1 = CalculateF1Score(gtImg = testY, binImg = binY)
         
         IoU_list.append(IoU);  F1_list.append(F1); prec_list.append(prec);  recall_list.append(recall)
         
         count += 1
    
    meanIoU = np.mean(IoU_list)
    meanF1  = np.mean(F1_list)
    meanPrec = np.mean(prec_list)
    meanRecall = np.mean(recall_list)


    print('Total number of test images is:', count)    
    print('The average statitisc measures are: ')
    print('IoU = ', round(meanIoU, 4), ' F1 = ', round(meanF1 ,4), 'prec = ', round(meanPrec, 4), 'recall = ', round(meanRecall, 4))

#Test a group of images (eg. from one dataset)
#Note:  1. All test images should be put under the same directory
#       2. All GT images should be put under the same directory
#       3. The TP, FP, FN, Recall, Precision, F1 score and IoU score will be calculated 
#       4. The stitched output from network will be saved in another directory
def twoStageTestImages(segNet, segNet2,
                       groupDir = '../data/MASS/TestX/', groupGTDir = '../data/MASS/TestY/',
                       nameSpliChar = '.',
                       gtImgType = '.tif',
                       saveDir  = '../output/SegResult/', sampleS = 512, stepW = -1, testMax = 1000, T_o = 5, isSave = True):
    
    if(stepW == -1):
        
        stepW = int(sampleS / 2)
    
    if(os.path.isdir(saveDir) == False):
            os.makedirs(saveDir)
    else:
        if(isSave):
            os.system('rm ' + saveDir + '*')
            
    fileList = os.listdir(groupDir) #Get the file names of the training images
    fileNum = min(len(fileList), testMax)
    
    count = 0
    IoU_list  = []; F1_list   = [];  prec_list = []; recall_list = [] #List for record the statistic results
    IoU2_list  = []; F12_list   = [];  prec2_list = []; recall2_list = [] #List for record the statistic results
    time_stage1 = 0
    time_stage2 = 0
    for i in range(fileNum):
         
         orgfile = fileList[i]
                 
         prefile = orgfile.split(nameSpliChar)[-2]
         
         gtfile = prefile + gtImgType

         gtfile = groupGTDir  + gtfile   #Get the ground truth image

         orgfile = groupDir + orgfile
         
         if(os.path.exists(gtfile) == False):
             continue
         
         
         print(i, 'Process image:', prefile)
         
         testX, testY = LoadTestImg(orgImgFile = orgfile, gtImgFile = gtfile, isLoadGT = True)
         
         #First stage segmentation 
         time0 = time.time()
         predY = segNet.SegStitch(testX, stepW = stepW, stepH = stepW, unitW = sampleS)
         time1 = time.time()
         time_stage1 += time1 - time0
#         predY = segNet.ExpandCut(testX)
         #Second stage segmentation
         testXR = np.zeros((testX.shape[0], testX.shape[1], testX.shape[2] + 1 ), dtype = 'uint')
         testXR[: , :, : testX.shape[2]] = testX
         testXR[:,  :, -1] = np.uint8(predY)
         
         time2 = time.time()
         predY2 = segNet2.SegStitch(testXR, stepW = stepW, stepH = stepW, unitW = sampleS)
         time3 = time.time()
         time_stage2 += time3 - time2
         
         ww = 1.0
         predY2 = ((ww * predY2.astype('uint') +(1 - ww) * predY)).astype('uint8')
         
#         if(isSave):
#         
#             saveName = saveDir + prefile
#             
#             
#             SaveShowImg(predY, testX, predImgName = saveName + '_stage1_pred.png', 
#                            visualImgName = saveName + '_stage1_vis.png' )
#             
#             SaveShowImg(predY2, testX, predImgName = saveName + '_stage2_pred.png', 
#                            visualImgName = saveName + '_stage2_vis.png' )
#             
#             SaveShowImg(testY, testX, predImgName = saveName + '_GT_pred.png', 
#                            visualImgName = saveName + '_GT_vis.png' ) 
         
         
#         
         #Save 1/4 images
#         SaveShowSub4Img(predY,  testX, saveName = saveName + '_stage1')
#         SaveShowSub4Img(predY2, testX, saveName = saveName + '_stage2')
#         SaveShowSub4Img(testY,  testX, saveName = saveName + '_GT')
        
#         np.save(saveDir + prefile + '_pred.npy', predY)
#         np.save(saveDir + prefile + '_pred2.npy', predY2)

         binY = BinarizedImg(predY, T = 128)
         binY2 = BinarizedImg(predY2, T = 128)
         
         if(T_o > 0 ):
             
             binY = opening(binY, disk(T_o))
             binY2 = opening(binY2, disk(T_o))
         
         
         testY = BinarizedImg(testY, T = 128)
         
         binY2 = binY2.astype('bool')
         binY = binY.astype('bool')
         testY = testY.astype('bool')
         
         if(isSave):
         
             saveName = saveDir + prefile         
             SaveRGB_FPTPFN(testX, binImg = binY, gtImg = testY, saveImgName = saveName + '_stage1_pred.png' )
             SaveRGB_FPTPFN(testX, binImg = binY2, gtImg = testY, saveImgName = saveName + '_stage2_pred.png' )
         
         print('Result for stage 1:')        
         IoU = CaculateIoUScore(gtImg = testY, binImg = binY)        
         prec, recall, F1 = CalculateF1Score(gtImg = testY, binImg = binY)
         
         print('Result for stage 2:')           
         IoU2 = CaculateIoUScore(gtImg = testY, binImg = binY2)        
         prec2, recall2, F12 = CalculateF1Score(gtImg = testY, binImg = binY2)
         
         IoU_list.append(IoU);  F1_list.append(F1); prec_list.append(prec);  recall_list.append(recall)
         IoU2_list.append(IoU2);  F12_list.append(F12); prec2_list.append(prec2);  recall2_list.append(recall2)         
         count += 1
    
    t_num = len(IoU_list)
    t_num4 = t_num * 4
    print('Avg stage1 prediction time is:', round(time_stage1/t_num, 2), round(time_stage1/t_num4, 2))
    print('Avg stage2 prediction time is:', round(time_stage2/t_num, 2), round(time_stage2/t_num4, 2) )
    print('Avg prediction time is:',round((time_stage2 + time_stage1)/t_num, 2), round((time_stage2 + time_stage1)/t_num4, 2))
    meanIoU = np.mean(IoU_list)
    meanF1  = np.mean(F1_list)
    meanPrec = np.mean(prec_list)
    meanRecall = np.mean(recall_list)

    meanIoU2 = np.mean(IoU2_list)
    meanF12  = np.mean(F12_list)
    meanPrec2 = np.mean(prec2_list)
    meanRecall2 = np.mean(recall2_list)


    print('Total number of test images is:', count)    
    print('The average statitisc measures for stage 1 are: ')
    print('IoU = ', round(meanIoU, 4), ' F1 = ', round(meanF1 ,4), 'prec = ', round(meanPrec, 4), 'recall = ', round(meanRecall, 4))

    print('The average statitisc measures for stage 2 are: ')
    print('IoU = ', round(meanIoU2, 4), ' F1 = ', round(meanF12, 4), 'prec = ', round(meanPrec2, 4), 'recall = ', round(meanRecall2, 4))




#Shuffle the training data   
def RandomShuffle(trainX, trainY, yDim = 1 ):
        
        
    print(trainX.shape, trainY.shape)
    train_s = np.concatenate((trainX, trainY), axis = 1) #Concatenate x,y before shuffle
        
    np.random.shuffle(train_s)
    
    trainX_n = train_s[ : , : -yDim ,  :]
    trainY_n = train_s[ : , -yDim : ,  :]
        
    return trainX_n, trainY_n


def RandomShuffle2(train_s, yDim = 1):
        
    np.random.shuffle(train_s)
        
    trainX_n = train_s[ : , : -yDim ,  :]
    trainY_n = train_s[ : , -yDim : ,  :]
        
    return trainX_n, trainY_n
#Randomly splits trainX, trainY into two parts, Feb 6th

def SplitTrainImgs(trainXDir  = '../data/DeepGlobe/trainXOrg/', trainYDir = '../data/DeepGlobe/trainYOrg/', 
                   ntrainXDir = '../data/DeepGlobe/TrainX/', ntrainYDir ='../data/DeepGlobe/TrainY/', 
                   ntestXDir  = '../data/DeepGlobe/TestX/',   ntestYDir ='../data/DeepGlobe/TestY/', 
                   nameSpliChar = '_',
                   gtPosix = '_mask.png',
                   splitRatio = 0.9):
    
    if(os.path.isdir(ntrainXDir) == False):
            os.makedirs(ntrainXDir) 
    else:
            os.system('rm ' + ntrainXDir + '*')            
            
    if(os.path.isdir(ntrainYDir) == False):
            os.makedirs(ntrainYDir)     
    else:
            os.system('rm ' + ntrainYDir + '*')

    if(os.path.isdir(ntestXDir) == False):
            os.makedirs(ntestXDir) 
    else:
            os.system('rm ' + ntestXDir + '*')

    if(os.path.isdir(ntestYDir) == False):
            os.makedirs(ntestYDir) 
    else: 
            os.system('rm ' + ntestYDir + '*')
        
    fileList = os.listdir(trainXDir) #Get the file names of the training images
    fileNum = len(fileList)
    
    trainCnt = 0
    testCnt  = 0
    
    if(splitRatio <= 1):
        trainNum = int(splitRatio * fileNum)
    else:
        trainNum = splitRatio
    
    orderList = [f for f in range(fileNum)]
    random.shuffle(orderList)
    
    print('Num of Org total train images:', fileNum)
    
    for i in orderList:
        
         orgfile = fileList[i]
         
         print(orgfile)
         
         prefile = orgfile.split(nameSpliChar)[0]
         
         gtfile = prefile + gtPosix

         orgX = trainXDir + orgfile
         orgY  = trainYDir + gtfile   #Get the ground truth image
        
         if(os.path.exists(orgY) == False):
             continue
         
         if(trainCnt < trainNum):
            
            dstX = ntrainXDir + orgfile
            dstY = ntrainYDir + gtfile
        
            os.system('cp ' + orgX + ' ' + dstX)
            os.system('cp ' + orgY + ' ' + dstY)  
            
            trainCnt +=1
            
         else:
             
            dstX = ntestXDir + orgfile
            dstY = ntestYDir + gtfile
            
            os.system('cp ' + orgX + ' ' + dstX)
            os.system('cp ' + orgY + ' ' + dstY)  
            
            testCnt +=1     
    
    print('Num of new train files:', trainCnt)
    print('Num of new test files', testCnt)





'''***********    Validate the IoU/F1 score by other people's code        *********'''
def mean_iou(truth_i, pred_i): #Calculate IoU by the same code fro  SDUNet paper

    union = np.sum(np.logical_or(pred_i, truth_i))
    intersection = np.sum(np.logical_and(pred_i, truth_i))
    iou = float(intersection) / (float(union)+0.001)
    
    return iou

def f1_score(truth, pred): #Calculate f1-score by the same code from SDUNet paper
        
    tp = np.logical_and(pred, truth)
    tn = np.logical_and(~pred, ~truth)
    
    tp_i = np.sum(tp)
    pred_i = np.sum(pred)
    
    truth_i = np.sum(truth) 
    tn_i = np.sum(tn)
    fn_i = np.sum(~pred) - tn_i
            # print(truth_i)
    recall_i = (float(tp_i)+0.001) / (float(truth_i)+0.001)
    
    if pred_i == 0:
        f1_score_i = 0
    else:
        precision_i = float(tp_i) / float(pred_i) + 1e-3
        f1_score_i = 2 * precision_i * recall_i / (precision_i + recall_i)
    
    iou = float(tp_i)/((float(pred_i) + fn_i))
   
    return precision_i, recall_i, f1_score_i, iou

'''***********   Functions may not be used        *************'''
 ###Convert 16 Uint image to 8 Uint image        
def CvtU16ToU8(imgun16):
     
    totalDims = imgun16.shape[2]
     
    imgun8 = np.zeros((imgun16.shape[0], imgun16.shape[1], totalDims), dtype = 'uint8')
     
    for dim in range(totalDims):
         
        imgun8[:, :, dim] = img_as_ubyte(exposure.rescale_intensity(imgun16[:, :, dim]))
    
    return imgun8
    
 ###Convert 16 Uint image to float 32 image        
def CvtImgToF32(orgImg):
     
    totalDims = orgImg.shape[2]
     
    imgF32 = np.zeros((orgImg.shape[0], orgImg.shape[1], totalDims), dtype = 'float32')
     
    for dim in range(totalDims):
         
        imgF32[:, :, dim] = img_as_float(exposure.rescale_intensity(orgImg[:, :, dim]))
    
    return imgF32
#SplitTrainImgs(trainXDir  = '../data/MASS/TrainX/', trainYDir = '../data/MASS/TrainY/', 
#                   ntrainXDir = '../data/MASS/TrainX1/', ntrainYDir = '../data/MASS/TrainY1/', 
#                   ntestXDir  = '../data/MASS/TrainX2/',   ntestYDir = '../data/MASS/TrainY2/', 
#                   nameSpliChar = '.',
#                   gtPosix = '.tif',
#                   splitRatio = 0.5)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
