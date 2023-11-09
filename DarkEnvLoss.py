import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

# color loss   eduction or distortion of color information in an image
class ColorLoss(nn.Module):

    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x ):

        # batch size, number of channels, height, width
        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)  # calculate the mean RGB values for the input image
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)  # split the mean RGB values into individual channels

        # calculate the color differences in the RGB channels
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)

        # compute the color loss as the Euclidean distance between color differences
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

# accurately localize
class SpatialLoss(nn.Module):

    def __init__(self):
        super(SpatialLoss, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # Define spatial convolution kernels
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)

        # Create convolutional kernels for spatial operations
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        # compute the mean of the original and enhanced images
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        # average pooling to the mean images
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        # Compute weight differences and enhance differences
        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)

        # calculate differences using convolution operations
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        # Calculate the spatial loss
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
    
class ExposureLoss(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(ExposureLoss, self).__init__()
        # Create an average pooling layer with a specified patch size
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        # Calculate the exposure loss
        b, c, h, w = x.shape

        # Compute the mean of the input image
        x = torch.mean(x, 1, keepdim=True)
        
        # Apply average pooling to the mean image
        mean = self.pool(x)
        
        # Calculate the exposure loss as the mean squared difference from the specified mean value
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d

        

# Total Variation (TV) Loss
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # Get the batch size and dimensions of the input
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        # Compute the count of horizontal and vertical neighbors
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)

        # Calculate the TV loss by computing differences between neighboring pixels
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        # Scale the TV loss by the specified weight and normalize by the batch size
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

# Structural and Amplitude Loss (Sa_Loss)
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        # Get the batch size and dimensions of the input
        b, c, h, w = x.shape
        
        # Split the input tensor into its RGB channels
        r, g, b = torch.split(x, 1, dim=1)
        
        # Calculate the mean RGB values for the input image
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        
        # Compute the differences in RGB channels
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        
        # Calculate the structural loss by measuring the Euclidean distance of RGB differences
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)

        # Compute the mean of the calculated structural loss
        k = torch.mean(k)
        return k











# class perception_loss(nn.Module):
#     def __init__(self):
#         super(perception_loss, self).__init__()
#         features = vgg16(pretrained=True).features
#         self.to_relu_1_2 = nn.Sequential() 
#         self.to_relu_2_2 = nn.Sequential() 
#         self.to_relu_3_3 = nn.Sequential()
#         self.to_relu_4_3 = nn.Sequential()

#         for x in range(4):
#             self.to_relu_1_2.add_module(str(x), features[x])
#         for x in range(4, 9):
#             self.to_relu_2_2.add_module(str(x), features[x])
#         for x in range(9, 16):
#             self.to_relu_3_3.add_module(str(x), features[x])
#         for x in range(16, 23):
#             self.to_relu_4_3.add_module(str(x), features[x])
        
#         # don't need the gradients, just want the features
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         h = self.to_relu_1_2(x)
#         h_relu_1_2 = h
#         h = self.to_relu_2_2(h)
#         h_relu_2_2 = h
#         h = self.to_relu_3_3(h)
#         h_relu_3_3 = h
#         h = self.to_relu_4_3(h)
#         h_relu_4_3 = h
#         # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
#         return h_relu_4_3
