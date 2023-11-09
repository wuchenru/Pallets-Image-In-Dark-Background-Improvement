import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class DarkModel(nn.Module):

	def __init__(self):
		super(DarkModel, self).__init__()

		# Number of features for convolutional layers
		features = 32

        # Convolutional Network

        # Starts with RGB channels, transforms it into feature channels, stride = 1(step/pixel), padding = 1 to match the size
		self.e_conv1 = nn.Conv2d(3, features, 3, 1, 1, bias=True) # 3x3 filter

        # features input channels, features output channels, 
		self.e_conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
		self.e_conv3 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
		self.e_conv4 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        
        # concatenated features from 3 and 4
		self.e_conv5 = nn.Conv2d(features * 2, features, 3, 1, 1, bias=True)

        # concatenated features from 2 and 5
		self.e_conv6 = nn.Conv2d(features * 2, features, 3, 1, 1, bias=True)

        # concatenated features from 1 and 6
		self.e_conv7 = nn.Conv2d(features * 2, 24, 3, 1, 1, bias=True)
        
        # activation function, ReLU commonly used
		self.relu = nn.ReLU(inplace=True)

        # max pooling and upsampling layers
		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Max Pooling: Reduces image size by taking the maximum value in 2x2 regions, simplifying the information.
        # Upsampling: Enlarges the image using bilinear interpolation with a scale factor of 2, preserving details.

	def forward(self, x):

        # Apply ReLU activation and convolutional layers
		x1 = self.relu(self.conv1(x))
		x2 = self.relu(self.conv2(x1))
		x3 = self.relu(self.conv3(x2))
		x4 = self.relu(self.conv4(x3))

        # Combine features from previous layers
		x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
		x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))

        # Generate the enhanced image
		x_r = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
		r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # Enhance the input image using the learned features
		x = x + r1 * (torch.pow(x, 2) - x)
		x = x + r2 * (torch.pow(x, 2) - x)
		x = x + r3 * (torch.pow(x, 2) - x)
		enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
		x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
		x = x + r6 * (torch.pow(x, 2) - x)
		x = x + r7 * (torch.pow(x, 2) - x)
		enhance_image = x + r8 * (torch.pow(x, 2) - x)
		r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)

        # Return the enhanced image, intermediate enhancement, and feature map
		return enhance_image_1, enhance_image, r



