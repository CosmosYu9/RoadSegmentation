# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""******************************************************************************************************
*  This is the code for generate the ResUNet network                                                    *
*  Latest update: Feb 1st, 2021                                                                         * 
*  By Cosmos&yu                                                                                         *
*  Define the neural networks: UNet, CUNet, DinkNet, ResUnet                                            *
*  The DinkNet34 is from https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge"              *
******************************************************************************************************"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial


#Parameters set the parameters of the networks
class Parameters:
    
    def __init__(self, inputDim = 3, unet_nfilters = 64, unet_dropout = 0.08, unet_layerNum = 6):
        
        self.unet_nfilters = unet_nfilters
        self.unet_dropout = unet_dropout
        self.unet_layerNum = unet_layerNum
        self.inputDim = inputDim
        
def getNetwork(network = 'unet', par = Parameters()):
    
        
        if(network == 'unet'):
            
            net = UNet(inputDim = par.inputDim, nfilters = par.unet_nfilters, dropout = par.unet_dropout, layerNum = par.unet_layerNum)

        elif(network == 'cunet'):
            
            net = CUNet(inputDim = par.inputDim, nfilters = par.unet_nfilters, dropout = par.unet_dropout, layerNum = par.unet_layerNum)

        elif(network == 'resunet'):
            
            net = ResUnet(channel =  par.inputDim)
            
        elif(network == 'dinknet34'):

            net = DinkNet34(num_channels = par.inputDim)
            
        return net

#2D Data:(BatchN, dim, H, W)
class UNet(nn.Module):
       
    def __init__(self, inputDim = 3, nfilters = 64, dropout = 0.08, layerNum = 6):
        
        print('This is UNet!')        
        print('UNet LayerNumer is:', layerNum)
        super(UNet, self).__init__()
        
        # Layers in contracting path  
        self.downLayers = nn.ModuleList() #The list to stack the encoder layers
        self.upLayers   = nn.ModuleList() #The list to stack the decoder layers
        self.tranLayers = nn.ModuleList()
        self.layerNum = layerNum
        
        for i in range(layerNum): #
            
            if(i == 0):
                
                self.downLayers.append(self.conv2d_block(inputDim, nfilters))
            else:
                
                self.downLayers.append(self.conv2d_block(nfilters * (2 ** (i - 1)), nfilters * (2 ** (i))))
            

                self.tranLayers.append(nn.ConvTranspose2d(in_channels = nfilters * (2 ** (layerNum - i)), 
                                                      out_channels = nfilters * (2 ** (layerNum - i - 1)), 
                                                      kernel_size = 4, stride = 2, padding = 1, bias = False))
                
                self.upLayers.append(self.conv2d_block(nfilters * (2 ** (layerNum - i)), nfilters * (2 ** (layerNum - i - 1)))) #tran||down + drop2 + up                
                     
        
        self.convLast = nn.Sequential(
                         nn.Conv2d(in_channels = nfilters, out_channels = 1, 
                                   kernel_size = 3, stride = 1, padding = 1 ),
                         nn.Sigmoid()
                         )
        
        self.drop1 = nn.Dropout(p = dropout * 0.5)
        self.drop2 = nn.Dropout(p = dropout)
        self.max_pool = nn.MaxPool2d(2)

    def conv2d_block(self, dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = True, batchNorm = True): #The 2D convolution  Residual block
        
        #Define first layer
        conv1 = nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        
        batch1 = nn.BatchNorm2d(dim_out)
        
        relu1 = nn.ReLU()
        
        #Define second layer
        conv2 = nn.Conv2d(dim_out, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        
        batch2 = nn.BatchNorm2d(dim_out)   
        
        relu2 = nn.ReLU()
        
        return nn.Sequential(conv1, batch1, relu1, conv2, batch2, relu2)
 
    def forward(self, x):
    
        #Contracting path
        c = []; p = []
        for i in range(self.layerNum - 1):
            
            if(i == 0):
              c.append(self.downLayers[i](x))
              p.append(self.drop1(self.max_pool(c[-1])))       
            else:
              c.append(self.downLayers[i](p[-1]))
              p.append(self.drop2(self.max_pool(c[-1])))
        
        #Middle Layer
        c.append(self.downLayers[self.layerNum - 1](p[-1]))
        u = [c[-1]]
        
        #Expansive path  
        for i in range(self.layerNum - 1):
            
            u.append(self.tranLayers[i](u[-1]))
            u[-1] = torch.cat((u[-1], c[-2 - i]), dim = 1)
            u[-1] = self.drop2(u[-1])
            u[-1] = self.upLayers[i](u[-1])
                 
        outputs = self.convLast(u[-1])
        
        return outputs
    
    
#2D Data:(BatchN, dim, H, W)
class CUNet(nn.Module):
       
    def __init__(self, inputDim = 3, nfilters = 64, dropout = 0.08, layerNum = 6):
        
        print('This is CUNet!')
        print('UNet LayerNumer is:', layerNum)
        super(CUNet, self).__init__()
        
        # Layers in contracting path  
        self.downLayers = nn.ModuleList() #The list to stack the encoder layers
        self.upLayers   = nn.ModuleList() #The list to stack the decoder layers
        self.tranLayers = nn.ModuleList()
        self.layerNum = layerNum
        
        for i in range(layerNum): #
            
            if(i == 0):
                
                self.downLayers.append(self.conv2d_block(inputDim, nfilters))
            else:
                
                self.downLayers.append(self.conv2d_block(nfilters * (2 ** (i - 1)), nfilters * (2 ** (i))))
            

                self.tranLayers.append(nn.ConvTranspose2d(in_channels = nfilters * (2 ** (layerNum - i)), 
                                                      out_channels = nfilters * (2 ** (layerNum - i - 1)), 
                                                      kernel_size = 4, stride = 2, padding = 1, bias = False))
                
                self.upLayers.append(self.conv2d_block(nfilters * (2 ** (layerNum - i)), nfilters * (2 ** (layerNum - i - 1)))) #tran||down + drop2 + up                
        
        self.convLast =  nn.Conv2d(in_channels = nfilters, out_channels = 1,  kernel_size = 3, stride = 1, padding = 1 )
        self.sigmoid = nn.Sigmoid()         
               
        self.drop1 = nn.Dropout(p = dropout * 0.5)
        self.drop2 = nn.Dropout(p = dropout)
        self.max_pool = nn.MaxPool2d(2)
        
    def conv2d_block(self, dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = True, batchNorm = True): #The 2D convolution  Residual block
        
        #Define first layer
        conv1 = nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        
        batch1 = nn.BatchNorm2d(dim_out)
        
        relu1 = nn.ReLU()
        
        #Define second layer
        conv2 = nn.Conv2d(dim_out, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        
        batch2 = nn.BatchNorm2d(dim_out)   
        
        relu2 = nn.ReLU()
        
        return nn.Sequential(conv1, batch1, relu1, conv2, batch2, relu2)
 
    def forward(self, x):
    
        #Contracting path
        c = []; p = []
        for i in range(self.layerNum - 1):
            
            if(i == 0):
              c.append(self.downLayers[i](x))
              p.append(self.drop1(self.max_pool(c[-1])))  
              p.append(self.max_pool(c[-1]))
            else:
              c.append(self.downLayers[i](p[-1]))
              p.append(self.max_pool(c[-1]))
              p.append(self.drop2(self.max_pool(c[-1])))
        
        #Middle Layer
        c.append(self.downLayers[self.layerNum - 1](p[-1]))
        u = [c[-1]]
        
        #Expansive path  
        for i in range(self.layerNum - 1):
            
            u.append(self.tranLayers[i](u[-1]))
            u[-1] = torch.cat((u[-1], c[-2 - i]), dim = 1)
            u[-1] = self.drop2(u[-1])
            u[-1] = self.upLayers[i](u[-1])
        
        outputs = self.convLast(u[-1]) + x[:, -1 :, :, :] #Short cut between output and mask
        
        outputs =  self.sigmoid(outputs)
        
        return outputs


"*****************************************************************************************************"
"DinkNet code from https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge"
nonlinearity = partial(F.relu,inplace=True)

class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        print('This is DinkNet34!')       
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights = 'DEFAULT')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

"*****************************************************************************************************"

class ResConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding = 1, kernel_size = 3):
        super(ResConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size = kernel_size, stride = stride, padding = padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size = kernel_size, padding = padding),
        )
            
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class ResUnet(nn.Module):
    def __init__(self, channel, filters=[128, 256, 512, 1024]):
        super(ResUnet, self).__init__()

        print('This is ResUnet!')
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size = 3, padding = 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size = 3, padding = 1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size = 3, padding = 1)
        )

        self.residual_conv_1 = ResConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResConv(filters[1], filters[2], 2, 1)

        self.bridge = ResConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = nn.ConvTranspose2d(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = nn.ConvTranspose2d(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = nn.ConvTranspose2d(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
    
"*****************************************************************************************************"


    
