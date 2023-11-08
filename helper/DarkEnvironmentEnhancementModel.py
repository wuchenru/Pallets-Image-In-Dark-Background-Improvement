import torch
import torch.nn as nn
import torch.nn.functional as F

class DarkEnvironmentEnhancementModel(nn.Module):

    def __init__(self):
        super(DarkEnvironmentEnhancementModel, self).__init()

        # activation function, ReLU commonly used
        self.relu = nn.ReLU(inplace=True)

        # Number of features for convolutional layers
        num_features = 32

        # Convolutional layers for enhancement
        
        # Conv1: First convolutional layer with 3 input channels, num_features output channels, 
        # 3x3 kernel, 1 stride, 1 padding, and bias
        self.conv1 = nn.Conv2d(3, num_features, 3, 1, 1, bias=True)

        # Conv2: Second convolutional layer with num_features input channels, num_features output channels, 
        # 3x3 kernel, 1 stride, 1 padding, and bias
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)

        # Conv3: Third convolutional layer with num_features input channels, num_features output channels, 
        # 3x3 kernel, 1 stride, 1 padding, and bias
        self.conv3 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)

        # Conv4: Fourth convolutional layer with num_features input channels, num_features output channels, 
        # 3x3 kernel, 1 stride, 1 padding, and bias
        self.conv4 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)

        # Conv5: Fifth convolutional layer with concatenated features from Conv3 and Conv4, 
        # num_features*2 input channels, num_features output channels, 3x3 kernel, 1 stride, 1 padding, and bias
        self.conv5 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias=True)

        # Conv6: Sixth convolutional layer with concatenated features from Conv2 and Conv5, 
        # num_features*2 input channels, num_features output channels, 3x3 kernel, 1 stride, 1 padding, and bias
        self.conv6 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias=True)

        # Conv7: Seventh convolutional layer with concatenated features from Conv1 and Conv6, 
        # num_features*2 input channels, 24 output channels (for enhancement), 
        # 3x3 kernel, 1 stride, 1 padding, and bias
        self.conv7 = nn.Conv2d(num_features * 2, 24, 3, 1, 1, bias=True)

        # Max pooling and upsampling layers
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):

        # Apply ReLU activation and convolutional layers
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        # Combine features from previous layers
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1))

        # Generate the enhanced image
        x_r = F.tanh(self.conv7(torch.cat([x1, x6], 1))
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
