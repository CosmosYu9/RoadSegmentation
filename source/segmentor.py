#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""************************************************************
*  Define the trainer (Segmentation) for neural networks      *
*  Latest update: Feb 1st, 2021                               *
*  By Cosmos&yu                                               *
************************************************************"""
import torch
import torch.nn as nn
import numpy as np
import os
import time
import math
import torch.utils.data as data

class ImageFolder(data.Dataset):
    
    def __init__(self, filelist, orgDir, gtDir, org_pos = '_sat.npy', gt_pos = '_mask.npy' ):
        
        self.ids = filelist
        self.orgDir = orgDir
        self.gtDir = gtDir

        self.org_pos = org_pos
        self.gt_pos = gt_pos

    def __getitem__(self, index):
        
        id = self.ids[index]
        
        img  = np.load(self.orgDir + id + self.org_pos) 
        mask = np.load(self.gtDir + id + self.gt_pos)
                
        return img, mask

    def __len__(self):
        return len(self.ids)
#torch.cuda.set_device(0)

#Dice bce loss class 
#This is loss defined from paper D-LinkNet: 
""" D-LinkNet:LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction """
class dice_bce_loss(nn.Module):
    
    def __init__(self, batch = True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
                
    def soft_dice_coeff(self, y_pred, y_true):
        
        smooth = 0.0  # may change
        
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true,):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss
        
    def __call__(self, y_pred,  y_true, w = 0.5):
        
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss( y_pred, y_true)

 #       return a + b     
        return a * w + b * (1 - w)
    
#Dice + Weight BCE; Give weights to BCE
class dice_loss(nn.Module):
    
    def __init__(self, batch = True):
        super(dice_loss, self).__init__()
        self.batch = batch
        
    def soft_dice_coeff(self, y_pred, y_true):
        
        smooth = 0.0  # may change
        
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
    
        return loss
        
    def __call__(self, y_pred, y_true):
        
        return self.soft_dice_loss(y_pred, y_true)

#Dice + Weight BCE; Give weights to BCE
class IoU_score(nn.Module):
    
    def __init__(self, batch = True):
        super(IoU_score, self).__init__()
        self.batch = batch
     
    def soft_dice_coeff(self, y_pred, y_true):
        
        smooth = 0.0000001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()
        
    def __call__(self, y_pred, y_true):
        
        return  self.soft_dice_coeff(y_pred, y_true)


class segmentor:
#width: width of input image
#height: height of input image
#inputDim: dimension of input image, eg. rgb: 3
#par: parameters of used network
    def __init__(self, network, inputDim = 3,  
                       epochs = 300, batch_size = 16, learn_rate = 0.0002, 
                       early_patience = 25, lr_patience = 6, lr_factor = 0.999, 
                       lossSel = 'BCE', useMultiGPU = False):
        
                #Early stopping, if the val loss not decrease over patience epochs, then stop learning
        self.early_stopping = EarlyStopping(patience = early_patience, min_delta = 0)
 
        #Select the network to be used
        self.network = network
            
        network_params = sum(p.numel() for p in self.network.parameters())
        network_train_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        print('The network parameters:', np.round(network_params / 1000000, 2), 'M')
        print('Trainable parameters:', np.round(network_train_params / 1000000, 2), 'M')
        
        
        #If cuda is available, select the cuda list    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #If there is gpu then select gpu
        print('Device is:', self.device)             
        if(self.device == torch.device('cuda')):
            if(useMultiGPU):
                self.network = torch.nn.DataParallel(self.network, range(torch.cuda.device_count()))
            print('Current GPU list:', range(torch.cuda.device_count()))
               
        self.network.to(self.device)  #using GPU
        
        self.epochs = epochs 
        
        self.batch_size = batch_size
        
        self.learn_rate = learn_rate
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learn_rate, amsgrad = True)
    
        self.diceBCE_loss =  dice_bce_loss() #Initialize the dice loss object
        self.dice_loss = dice_loss()
        self.IoU_score = IoU_score()
        self.bce_loss = nn.BCELoss()
#Select loss function used for training
        if(lossSel == 'BCE'):
            self.LossFunc = self.LossBCE
        elif(lossSel == 'WBCE'):
            self.LossFunc = self.LossWBCE
        elif(lossSel == 'DBCE'):
            self.LossFunc = self.LossDBCE
        elif(lossSel == 'MSE'):
            self.LossFunc = self.LossMSE
        elif(lossSel == 'DWBCE'):
            self.LossFunc = self.LossDWBCE
        
        self.lossSel = lossSel
        
        self.losses = []
        
        self.valLosses = []

        #Set the learning rate scheuler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode = 'min',
                patience = lr_patience, #if the loss feed into scheuler not decrease over patience epochs, 
                               #then learning rate*factor
                factor = lr_factor, #Decay rate, lr = lr * factor once the patience is meet
                min_lr = 1e-7,
                verbose = True  #When update is made then print the message
            ) #Scheduler is used to decrease the learning rate according to learning result


        
    def __del__(self):
        
        self.ReleaseGPU()
        
    def ReleaseGPU(self):
        torch.cuda.empty_cache() 
        
#Loading the trained models
    def Loading(self, fileName = 'UNet', fileDir = '../models/', best = False): 
        
        if(best == 1): #Load the trained network with the best validation loss
            file = fileDir + fileName + '_best_segmentor.pth'
        elif(best == 0): #Load the trained transformer with max training epochs
            file = fileDir + fileName + '_segmentor.pth'
        else:
            file = fileDir + fileName + '_IoU_segmentor.pth'
            
        if(os.path.exists(file)):
             self.network.load_state_dict(torch.load(file))
             print('loading ', file, ' successfully!')
             return True
        else:
            print('no training model!')
            return False


    def LossMSE(self, net_out, tensor_y, ets = 1e-15, w = 0.6):  #Mean Square Error
        
        return torch.mean((net_out - tensor_y) ** 2)

    def LossBCE(self, net_out, tensor_y, ets = 1e-15, w = 0.5): #Original BCE
        
        return self.bce_loss(net_out, tensor_y)

    
    def LossDBCE(self, net_out, tensor_y, ets = 1e-15, w = 0.5):
        
        return self.diceBCE_loss(net_out, tensor_y, w)
    
    def LossWBCE(self, net_out, tensor_y, ets = 1e-15, w = 0.6): #
        
        tempV = tensor_y * torch.log(net_out + ets) + (1 - tensor_y) * torch.log(1 - net_out + ets)
        tempV[tempV < -100] = -100   
               
        return -torch.mean(torch.mul(tempV, self.tensor_wMap))

#0.515466~ 0.97;  0.526316~0.95; 0.555556 ~  0.9
    def LossDWBCE(self, net_out, tensor_y, ets = 1e-15, w = 0.515466): #W=0.5 best Feb17, 0.555556 ~  0.9 theoretical good?
        
        a = self.LossWBCE(net_out, tensor_y)
        b = self.dice_loss(net_out, tensor_y)
        
        return a * w + b * 0.5

    def Validation(self, valX, valY):

        yDim = valY.shape[1]
        
        nSamples = valX.shape[0] #Total number of valiation data
        
        nBatches = int(nSamples / self.batch_size) #total batches
            
        elosses = []
        IoU_L = []
        for b in range(nBatches):
                
            X = valX[b * self.batch_size : (b + 1) * self.batch_size, :]
                
            Y = valY[b * self.batch_size : (b + 1) * self.batch_size, :] #0 means only predict next one! if want predict more, then need to change 
                
            if(yDim > 1): #Exsits a weight map for calculate more complex loss function
                    
                weightMap = Y[:, 1 : 2, : , :]
                    
                Y =         Y[:, 0 : 1, : , :]
                    
                self.tensor_wMap = torch.Tensor(weightMap / 255.0).to(self.device) #Weight map must be stored as uint8, [0.0, 1.0] -> [0,255]
            
            tensor_x = torch.Tensor(X / 255.0).to(self.device) #Load the X, Y to GPU
            tensor_y = torch.Tensor(Y / 255.0).to(self.device)             
            
            #Forward pass and calculate loss
            net_out = self.network(tensor_x)
            IoU = self.IoU_score(net_out, tensor_y)
            loss = self.LossFunc(net_out, tensor_y)
            #Return the validation loss for each channel separately
            eloss = loss.mean().item()
            IoU_L.append(IoU.mean().item())
            elosses.append(eloss)
        
        val_loss = np.mean(elosses)
        mIoU = np.mean(IoU_L)
        print('IoU: ', mIoU)
        
        return val_loss, mIoU
#Training sampleX, sampleY must be uint8
    def Training(self, sampleX, sampleY, ratio = 0.96, fileDir = '../models/', fileName = 'UNet', r = 5, div = 10):
        #data dimension: (num, dim, H, W)
     
        self.network.train()  #Back to train model
        
        
        print('Save model will be:' + fileDir + fileName + '_segmentor.pth')
        
        trainHistory = []  #Save the training loss, val loss, val iou for each epoch
        
        print('sampleX dimension:', sampleX.shape, 'Type:', sampleX.dtype, 'Max:', np.max(sampleX))
        print('sampleY dimension:', sampleY.shape, 'Type:', sampleY.dtype, 'Max:', np.max(sampleY))
        
        print('Begining training with ', self.lossSel, ' loss function')
        
        best_val__loss = 100000.0; best_IoU = 0
        
        yDim = sampleY.shape[1]
        
        time_s = time.time()
        
        trainBatches = int(np.ceil(sampleX.shape[0] / self.batch_size))
               
        trNums = int(trainBatches * ratio)
    
#       sampleX, sampleY = self.RandomShuffle(sampleX, sampleY, yDim = yDim) #Feb14 shuffle in generating process, keep val be the same
        
        trainX = sampleX[ : self.batch_size * trNums]
        trainY = sampleY[ : self.batch_size * trNums]
        
        valX = sampleX[self.batch_size * trNums : self.batch_size * trainBatches ]
        valY = sampleY[self.batch_size * trNums : self.batch_size * trainBatches, 0 : 2]
                 
        oneTenNum = int(trNums / div)
        
        trainS = np.concatenate((trainX, trainY), axis = 1) 
        for e in range(self.epochs): #training loop,maximum training epochs is self.epochs
             
            trainX, trainY = self.RandomShuffle2(trainS, yDim = yDim)
            elosses = []
            print('Epoch', e, end = "[", flush = True)
            time_b = time.time() 
            for b in range(trNums):
                
                if(b % oneTenNum == 0):
                    print('-', end = "", flush = True)
                
                self.optimizer.zero_grad()
            
                #Extract the bth batch data
                X = trainX[b * self.batch_size : (b + 1) * self.batch_size, :]
                
                Y = trainY[b * self.batch_size : (b + 1) * self.batch_size, :] #0 means only predict next one! if want predict more, then need to change 
                  
                if(yDim > 1): #Exsits a weight map for calculate more complex loss function
                    
                    weightMap = Y[:, 1 : 2, : , :]
                    
                    Y =         Y[:, 0 : 1, : , :]
                    
                    self.tensor_wMap = torch.Tensor(weightMap / 255.0).to(self.device) #Weight map must be stored as uint8, [0.0, 1.0] -> [0,255]
                
                tensor_x = torch.Tensor(X / 255.0).to(self.device) #Load the X, Y to GPU
                tensor_y = torch.Tensor(Y / 255.0).to(self.device)
                
                #Forward pass and calculate loss
                net_out = self.network(tensor_x)

                loss = self.LossFunc(net_out, tensor_y)  #Current the loss for current batch data

                # averages GPU-losses and performs a backward pass
                loss = loss.mean()
                #backwar pass
                loss.backward()

                self.optimizer.step()
                #Tracking losses 
                elosses.append(loss.item())
            
            train_epoch_loss = np.mean(elosses) #Mean loss in current epoch
            
            if(e % 10 == 0):
                torch.save(self.network.state_dict(),  fileDir + fileName + '_segmentor.pth') #Save final trained model

            #Test on validation data: sep_epoch_loss the loss for each channel
            val_epoch_loss, val_IoU = self.Validation(valX, valY)
            
            # if(val_IoU > best_IoU):
                
            #     best_IoU = val_IoU
            #     torch.save(self.network.state_dict(),  fileDir + fileName + '_IoU_segmentor.pth') #Save the model with the best val loss
            
            if(val_epoch_loss < best_val__loss):
                
                best_val__loss = val_epoch_loss
                
                torch.save(self.network.state_dict(),  fileDir + fileName + '_best_segmentor.pth') #Save the model with the best val loss
            
            print(']', end = " ", flush = True)
            print('train_losses = ', round(train_epoch_loss, r), 'val_losses = ', round(val_epoch_loss, r), 'Time:', round(time.time() - time_b, 2))
            
            trainHistory.append([train_epoch_loss, val_epoch_loss, val_IoU])
            
            time_e = time.time()
            
            print('total training time is:', round(time_e - time_s, 2))

            self.early_stopping(val_epoch_loss) #check the valiation loss for early stopping


            if self.early_stopping.early_stop: #If early_stop reached then break
                break
            
            self.lr_scheduler.step(train_epoch_loss) #scheduler check the train loss
            

          
        if(os.path.isdir(fileDir + 'trainlog/') == False):
            os.makedirs(fileDir + 'trainlog/') 
        
        np.save(fileDir + 'trainlog/' + fileName + '_histlog.npy', np.array(trainHistory))
            
        torch.save(self.network.state_dict(),  fileDir + fileName + '_segmentor.pth') #Save final trained model
                
#Use multi stage to do the segmentation    
    def multiPred(self, inputPatch, unitW = 512, M = 1):
        
        if(inputPatch.shape[1] == 3):  #Only rgb, then M = 1
            
            tensor_x = torch.Tensor(inputPatch / 255.0).to(self.device)
            
            net_out = self.network(tensor_x)
                
            outputPatch = net_out.cpu().detach().numpy().reshape(unitW, unitW)            
        
        else: #rgb + y_stage1
            
            inputNorm = inputPatch / 255.0
            
            for m in range(M):
                
                tensor_x = torch.Tensor(inputNorm).to(self.device)
                
                net_out = self.network(tensor_x)
                
                outputPatch = net_out.cpu().detach().numpy().reshape(unitW, unitW)            
                
                inputNorm [0, 3] = outputPatch
        
        return outputPatch
        
    def SegStitch(self, inputImg, stepW = 256, stepH = 256, unitW = 512, M = 2): #inputImg must be uint8
        
        self.network.eval()  #Move to evaluation model, don't want the random result due to dropout
        
        #inputimage shape: (width, height, dim)
        height = inputImg.shape[0]
        width  = inputImg.shape[1]
        
    
        if(unitW == height and unitW == width):

            inputPatch = inputImg
            inputPatch = np.expand_dims(inputPatch, axis = 0)
            inputPatch = np.swapaxes(inputPatch, axis1 = 1, axis2 = 3)            

            tensor_x = torch.Tensor(inputPatch / 255.0).to(self.device)
                
            net_out = self.network(tensor_x)
                    
            outputPatch = net_out.cpu().detach().numpy().reshape(unitW, unitW) 
                
            outputPatch = np.swapaxes(outputPatch, axis1 = 0, axis2 = 1) 
            
            return np.uint8(outputPatch * 255)  
       
        outputSeg = np.zeros((height, width), dtype = 'float32')
        outCnt = np.zeros((height, width), dtype = np.uint8)
        hsteps = math.ceil((height - unitW) / stepH) + 1
        wsteps = math.ceil((width - unitW) / stepW) + 1
        
        
        for h in range(hsteps):
            for w in range(wsteps):
    
                sH = stepH
                sW = stepW
                
                if(h==0):
                    sH = 0
                if(w==0):
                    sW = 0 
    
                upLeftY = h * sH
                upLeftX = w * sW
                
                if(upLeftY + unitW >= height):
                    upLeftY = height - unitW
                    sH = upLeftY - (h - 1) * stepH
                
                if(upLeftX + unitW >= width):
                    upLeftX = width - unitW
                    sW = upLeftX - (w - 1) * stepW
    
                inputPatch = inputImg[upLeftY : upLeftY + unitW, upLeftX : upLeftX + unitW,  :]      
                inputPatch = np.expand_dims(inputPatch, axis = 0)
                inputPatch = np.swapaxes(inputPatch, axis1 = 1, axis2 = 3)
                
#                outputPatch = self.multiPred(inputPatch = inputPatch, unitW = unitW, M = M)
                tensor_x = torch.Tensor(inputPatch / 255.0).to(self.device)
                
                net_out = self.network(tensor_x)
                    
                outputPatch = net_out.cpu().detach().numpy().reshape(unitW, unitW) 
                
                outputPatch = np.swapaxes(outputPatch, axis1 = 0, axis2 = 1)
                
                devh = int(sH / 2)
                devw = int(sW / 2)

                outputSeg[upLeftY + devh : upLeftY + unitW, upLeftX + devw : upLeftX + unitW] += outputPatch[devh : unitW, devw : unitW]
                outCnt[upLeftY + devh : upLeftY + unitW, upLeftX + devw : upLeftX + unitW] += 1
        
        outputSeg = outputSeg / outCnt
                
        return np.uint8(outputSeg * 255)  

    def RandomShuffle2(self,train_s, yDim = 1):
        
        np.random.shuffle(train_s)
        
        trainX_n = train_s[ : , : -yDim ,  :]
        trainY_n = train_s[ : , -yDim : ,  :]
        
        return trainX_n, trainY_n

    def ValidationD(self, valIter):

        elosses = []
        IoU_L = []
        
        for X, Y in valIter:
                
            yDim = Y.shape[1]
            
            if(yDim > 1): #Exsits a weight map for calculate more complex loss function
                    
                weightMap = Y[:, 1 : 2, : , :]
                    
                Y =         Y[:, 0 : 1, : , :]
                    
                self.tensor_wMap = torch.Tensor(weightMap / 255.0).to(self.device) #Weight map must be stored as uint8, [0.0, 1.0] -> [0,255]
            
            tensor_x = torch.Tensor(X / 255.0).to(self.device) #Load the X, Y to GPU
            tensor_y = torch.Tensor(Y / 255.0).to(self.device)             
            
#            X = None; del X; Y = None; del Y
            
            #Forward pass and calculate loss
            net_out = self.network(tensor_x)
            IoU = self.IoU_score(net_out, tensor_y)
            loss = self.LossFunc(net_out, tensor_y)
            #Return the validation loss for each channel separately

            eloss =  loss.mean().cpu().detach().numpy()
            iou = IoU.mean().cpu().detach().numpy()
#            eloss = loss.mean().item()            
#            IoU_L.append(IoU.mean().item())
            IoU_L.append(iou)
            elosses.append(eloss)
        
        val_loss = np.mean(elosses)
        mIoU = np.mean(IoU_L)
        
        return val_loss, mIoU
    
    def TrainingD(self, sampleDir, orgDir, gtDir, org_pos = '_sat.npy', gt_pos = '_mask.npy',
                        ratio = 0.96, fileDir = '../models/', fileName = 'UNet', r = 5, div = 10, num_workers = 0):

         
        allList = os.listdir(sampleDir)
        sampleList = [f.split('_')[0] for f in allList if f.find('sat') != -1]  
        
        self.network.train()  #Back to train model
        
        print('Save model will be:' + fileDir + fileName + '_segmentor.pth')
        
        trainHistory = []  #Save the training loss, val loss, val iou for each epoch
        totalSamples = len(sampleList)
        
        print('the number of samples is:' , totalSamples)
        np.random.shuffle(sampleList)  #Shuffle the samples before split them into train and val datasets
        

        print('Begining training with ', self.lossSel, ' loss function')
        
        best_val__loss = 100000.0; best_IoU = 0
                
        time_s = time.time()
        
        trainBatches = int(np.ceil(totalSamples / self.batch_size))
               
        trNums = int(trainBatches * ratio)
        

        trainList = sampleList[ : self.batch_size * trNums]
        valList   = sampleList[self.batch_size * trNums : ]
      
        trainSet = ImageFolder(trainList, orgDir, gtDir, org_pos = org_pos, gt_pos = gt_pos)
        valSet = ImageFolder(valList, orgDir, gtDir, org_pos = org_pos, gt_pos = gt_pos)
        
        
        trainLoader = data.DataLoader(trainSet, batch_size = self.batch_size, shuffle = True, num_workers = num_workers)
        valLoader   = data.DataLoader(valSet, batch_size = self.batch_size, shuffle = False, num_workers = num_workers)
        
                 
        oneTenNum = int(trNums / div)

        
        for e in range(self.epochs): #training loop,maximum training epochs is self.epochs
            
            b = 0
            trainIter = iter(trainLoader)
            valIter = iter(valLoader)
            elosses = []
            print('Epoch', e, end = "[", flush = True)
            time_b = time.time() 
            
            for X, Y in trainIter:
                if(b % oneTenNum == 0):
                    print('-', end = "", flush = True)
                
                b += 1
                
                self.optimizer.zero_grad()
            
                #Extract the bth batch data
                yDim = Y.shape[1]
                              
                if(yDim > 1): #Exsits a weight map for calculate more complex loss function
                    
                    weightMap = Y[:, 1 : 2, : , :]
                    
                    Y =         Y[:, 0 : 1, : , :]
                    
                    self.tensor_wMap = torch.Tensor(weightMap / 255.0).to(self.device) #Weight map must be stored as uint8, [0.0, 1.0] -> [0,255]
                
                tensor_x = torch.Tensor(X / 255.0).to(self.device) #Load the X, Y to GPU
                tensor_y = torch.Tensor(Y / 255.0).to(self.device)
                
#                X = None; del X; Y = None; del Y
                #Forward pass and calculate loss
                net_out = self.network(tensor_x)

                loss = self.LossFunc(net_out, tensor_y)  #Current the loss for current batch data

                # averages GPU-losses and performs a backward pass
                loss = loss.mean()
                #backwar pass
                loss.backward()

                self.optimizer.step()
                #Tracking losses 
                loss_val = loss.cpu().detach().numpy()
                elosses.append(loss_val)
#                elosses.append(loss.item())
            
            train_epoch_loss = np.mean(elosses) #Mean loss in current epoch
            
            if(e % 5 == 0):
                torch.save(self.network.state_dict(),  fileDir + fileName + '_segmentor.pth') #Save final trained model

            #Test on validation data: sep_epoch_loss the loss for each channel
           
            val_epoch_loss, val_IoU = self.ValidationD(valIter)
            
            if(val_IoU > best_IoU):
                
                best_IoU = val_IoU
                torch.save(self.network.state_dict(),  fileDir + fileName + '_IoU_segmentor.pth') #Save the model with the best val loss
            
            if(val_epoch_loss < best_val__loss):
                
                best_val__loss = val_epoch_loss
                
                torch.save(self.network.state_dict(),  fileDir + fileName + '_best_segmentor.pth') #Save the model with the best val loss
            
            print(']', end = " ", flush = True)
         
            print('train_losses = ', round(train_epoch_loss, r), 'val_losses = ', round(val_epoch_loss, r), 
                  'val_IoU = ', round(val_IoU, r),
                  'Time:', round(time.time() - time_b, 2))
            
            trainHistory.append([train_epoch_loss, val_epoch_loss])
            
            time_e = time.time()
            
            print('total training time is:', round(time_e - time_s, 2))

            self.early_stopping(val_epoch_loss) #check the valiation loss for early stopping


            if self.early_stopping.early_stop: #If early_stop reached then break
                break
            
            self.lr_scheduler.step(train_epoch_loss) #scheduler check the train loss
            

          
        if(os.path.isdir(fileDir + 'trainlog/') == False):
            os.makedirs(fileDir + 'trainlog/') 
        
        np.save(fileDir + 'trainlog/' + fileName + '_histlog.npy', np.array(trainHistory))
            
        torch.save(self.network.state_dict(),  fileDir + fileName + '_segmentor.pth') #Save final trained model

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience = 10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        
        if self.best_loss == None:
            
            self.best_loss = val_loss
            
        elif self.best_loss - val_loss > self.min_delta:
            
            self.best_loss = val_loss
            self.counter = 0
            
        elif self.best_loss - val_loss < self.min_delta:
            
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            
            if self.counter >= self.patience:
                
                print('INFO: Early stopping')
                self.early_stop = True

