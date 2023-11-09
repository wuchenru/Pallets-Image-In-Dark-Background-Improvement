import torch
import torch.nn as nn
import torch.nn.functional as F

class DarkEnvironmentEnhancementModel(nn.Module):

    def __init__(self):
        super(DarkEnvironmentEnhancementModel, self).__init__()

        features = 32

        self.conv1 = nn.Conv2d(3, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(features * 2, features, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(features * 2, features, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(features * 2, 24, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1))
        x_r = F.tanh(self.conv7(torch.cat([x1, x6], 1))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r
