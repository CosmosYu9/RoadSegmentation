"""************************************************************
*  Test Demos                                                 *
*  Latest update: Feb 1st, 2022                               *
*  By Cosmos&yu                                               *
*  This file used to test the 2-Stage training method         *
************************************************************"""
import genSamples as genS
import segmentor as seg
import utils as ut
import networks as nt #User can define new network in networks.py file
import time

GEN_SAMPLE_1STAGE = True #Generate random training samples for stage 1's network
TRAIN_1STAGE = True #Training the neural network in stage one
TEST_G_SEG = False  #True #Test semantic segmentation on a group of data saved as image in a directory

GEN_SAMPLE_2STAGE = True #Generate random training samples for stage 2's network
TRAIN_2STAGE = True #Training the neural network in stage two
TEST_G2_SEG = True #Test semantic segmentation
'''*******************************************************************************************************************'''
#Definition of Variables 
rootDir = '../' #Set the root directory of the project of RoadSegmentator

gtImgDict ={'MASS': '.tif', 'DeepGlobe': '.png'} #Set the format of the original images in the dataset                   
nameSplitDict = {'MASS': '.', 'DeepGlobe': '_'}

sID = 5 #The number of test (all saved files are related to this number)
dataset = 'MASS'

sampleSize = 512 #the training sample size: sampleSize * sampleSize
alpha = '002'#0xx: 0.xx, threshold to select key pixels for preparing training sample in stage 2
ap = list(alpha); ap.insert(1,'.'); 
delta = float(''.join(ap))

network1 = 'dinknet34' #select network
lossSel1 = 'DBCE' #select loss function
modelName1 = dataset + '_' + network1 + '_' + lossSel1 # the model Name to save the training neural network

network2 = 'cunet'
lossSel2 = 'DWBCE'
modelName2 = dataset + '_' + network1 + '_' + network2 + '_' + lossSel2 + alpha

#Variables about data preparation
gtImgDict ={'MASS': '.tif', 'DeepGlobe': '.png'} #Set the format of the original images in the dataset                   
nameSplitDict = {'MASS': '.', 'DeepGlobe': '_'} #for a pair of training image and test image: before the split sign, they should have same name
nameSpliChar = nameSplitDict[dataset]     #'.' for MASS
gtPosix   =  gtImgDict[dataset]           #'.tif' for MASS

valOrTest = 'Test' #Test on the test dataset: valOrTest + 'X' and  valOrTest + 'Y'

TrainXDir = rootDir + 'data/' + dataset + '/TrainX/'      #'../data/MASS/TrainX/'
TrainYDir = rootDir + 'data/' + dataset + '/TrainY/'  #'../data/MASS/TrainY/'
testXDir  = rootDir + 'data/' + dataset + '/' + valOrTest + 'X/'
testYDir  = rootDir + 'data/' + dataset + '/' + valOrTest + 'Y/'

sampleDir1 = rootDir + 'data/' + dataset + '/TrainSamples_SC' + str(sID) + '/'  #Save the extracted training samples into this file directory
sampleDir2 = rootDir + 'data/' + dataset + '/TrainSamples_2SC' + str(sID) + '/'  #Save the extracted training samples into this file directory
sampleDir22 = rootDir + 'data/' + dataset + '/TrainSamples_2SC' + str(sID) + '_' + modelName1 + alpha + '/'

saveAsMat = False          #True: save all training samples in one numpy MAT; False: save each sample as a pair of png images
maxNum = 2000000 #Upper limit of training samples

'''*******************************************************************************************************************'''
#Define the neural network in stage one
net1 = nt.getNetwork(network = network1, par = nt.Parameters(inputDim = 3))
segNet = seg.segmentor(network = net1, inputDim = 3,  
                       epochs = 100, batch_size = 16, learn_rate = 0.0001, 
                       early_patience = 10, lr_patience = 5, lr_factor = 0.99999, 
                       lossSel = lossSel1, useMultiGPU = False)
#Define the neural network in stage two
net2 = nt.getNetwork(network = network2, par = nt.Parameters(inputDim = 4, unet_nfilters = 32, unet_dropout = 0.01, unet_layerNum = 7))
segNet2C = seg.segmentor(network = net2, inputDim = 4,  
                       epochs = 100, batch_size = 8, learn_rate = 0.0001, 
                       early_patience = 10, lr_patience = 5, lr_factor = 0.9999999, 
                       lossSel = lossSel2, useMultiGPU = False)

'''*******************************************************************************************************************'''
if GEN_SAMPLE_1STAGE:
    K = [1, 1, 1, 1, 1]  #Frequency of each random augmentation operations 
    rotate = 8 
    #K[0]: randomly select K[0] training samples from trainX image
    #K[1], K[2]: flip trainX image from x, y axis respectively, then randomly select K[1] and K[2] training samples
    #K[3]: randomly darking the trainX, then randomly select K[3] training samples
    #K[4]: randomly rotate trainX image rotate times, then randomly select K[4] training samples from each rotated image
    #The function generate the training samples, generated samples are saved in saveDir
    genS.genTrainS4Seg(orgImgDir = TrainXDir , GTDir = TrainYDir, nameSpliChar = nameSpliChar, saveDir = sampleDir1, 
                           gtPosix = gtPosix, maxNum = maxNum, K = K, rotate = rotate, sampleSize = sampleSize, 
                           saveAsMat = saveAsMat)

'''*******************************************************************************************************************'''
if TRAIN_1STAGE:
    print('Training for Stage 1 with model ', modelName1, 'on Data ', sampleDir1) 
    #segNet.Loading(fileName = modelName1, fileDir = rootDir + 'models/', best = 0) #load previous trained network
    segNet.TrainingD(sampleDir = sampleDir1, orgDir = sampleDir1, gtDir = sampleDir1, 
                         org_pos = '_sat.npy', gt_pos = '_mask.npy',
                         ratio = 0.95, fileDir = rootDir + 'models/', fileName = modelName1, div =  10)
    
'''*******************************************************************************************************************'''
if TEST_G_SEG:
    saveDir  = rootDir + 'output/SegResult_' + dataset + '_' + modelName1 + '/'
    segNet.Loading(fileName = modelName1, fileDir = rootDir + 'models/', best = 1) #best = 1 #Use the network with best validation loss
    ut.testGroupImages(segNet = segNet, groupDir = testXDir, groupGTDir = testYDir,
                        nameSpliChar = nameSpliChar, gtImgType = gtPosix,
                        saveDir  = saveDir, sampleS = sampleSize, stepW = -1, testMax = 10000)

'''*******************************************************************************************************************'''
if GEN_SAMPLE_2STAGE:
    print('Generate 2nd Stage samples from ', sampleDir2, ' with ', modelName1, ' save in ', sampleDir22)
    K = [1, 1, 1, 1, 1]      # We can set different augmentation operations
    rotate = 12 
    genS.genTrainS4Seg(orgImgDir = TrainXDir , GTDir = TrainYDir, nameSpliChar = nameSpliChar, saveDir = sampleDir2, 
                            gtPosix = gtPosix, maxNum = maxNum, K = K, rotate = rotate, sampleSize = sampleSize, 
                            saveAsMat = saveAsMat)
        
    segNet.Loading(fileName = modelName1, fileDir = rootDir + 'models/', best = 0)  #Load the stage one nerual network
    genS.genRaw2ndStageTrainSamplesD(segNet, sampleDir2,  orgDir = sampleDir2, gtDir = sampleDir2, saveDir = sampleDir22, 
                                    org_pos = '_sat.npy', gt_pos = '_mask.npy', XRName = 'SampleX_Raw', YRName = 'SampleY_Raw',
                                    netDir = rootDir + 'models/', sampleS = sampleSize, div = 20, testNum = 200000, 
                                    wM = 1.0, wElse = 0.97, delta = delta, T_IoU = 0.3, saveAsMat = saveAsMat)

'''*******************************************************************************************************************'''
if TRAIN_2STAGE:
    print('Training for Stage 2 with model ', modelName2, 'on Data ', sampleDir22)
    # segNet2C.Loading(fileName = modelName2, fileDir = rootDir + 'models/', best = 0)  #load previous trained network
    segNet2C.TrainingD(sampleDir = sampleDir22, orgDir = sampleDir22, gtDir = sampleDir22, 
                     org_pos = '_sat.npy', gt_pos = '_mask.npy',
                     ratio = 0.95, fileDir = rootDir + 'models/', fileName = modelName2, div = 10 )
    
'''*******************************************************************************************************************'''
if TEST_G2_SEG:
    print('Test 2 Stage segmentation')
    saveDir  = rootDir + 'output/SegResult_' + dataset + '_' + modelName1 + '_' + modelName2 + '/'   
    isSave = False #True, then save the test results; False: only calculate the evluation metrics
    segNet.Loading(fileName = modelName1, fileDir = rootDir + 'models/', best = 0)
    segNet2C.Loading(fileName = modelName2, fileDir = rootDir + 'models/', best = 1)
    time_s = time.time()
    ut.twoStageTestImages(segNet = segNet, segNet2 = segNet2C,
                           groupDir = testXDir, groupGTDir = testYDir,
                           nameSpliChar = nameSpliChar,
                           gtImgType = gtPosix,
                           saveDir  = saveDir, sampleS = sampleSize, stepW = -1, testMax = 10000,  T_o = 0, isSave = isSave)
        
    print('Total test time is:', round(time.time() - time_s), 2)
