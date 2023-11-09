import torch
import torch.nn as nn
import torch.nn.functional as F

class DarkEnvironmentEnhancementLoss(nn.Module):

    def __init__(self):
        super(DarkEnvironmentEnhancementLoss, self).__init__()

        self.color_loss = ColorEnhancementLoss()
        self.spatial_loss = SpatialLoss()
        self.exposure_loss = ExposureLoss()
        self.tv_loss = TotalVariationLoss()
        self.saliency_loss = SaliencyLoss()
        self.perception_loss = PerceptionLoss()

    def forward(self, input_image, enhanced_image):
        color_loss = self.color_loss(enhanced_image)
        spatial_loss = self.spatial_loss(input_image, enhanced_image)
        exposure_loss = self.exposure_loss(enhanced_image)
        tv_loss = self.tv_loss(enhanced_image)
        saliency_loss = self.saliency_loss(enhanced_image)
        perception_loss = self.perception_loss(enhanced_image)

        total_loss = color_loss + spatial_loss + exposure_loss + tv_loss + saliency_loss + perception_loss

        return enhanced_image, total_loss

class ColorEnhancementLoss(nn.Module):

    def __init__(self):
        super(ColorEnhancementLoss, self).__init__()

    def forward(self, x):
        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k

class SpatialLoss(nn.Module):

    def __init__(self):
        super(SpatialLoss, self).__init()
        self.left_kernel = nn.Parameter(data=torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.right_kernel = nn.Parameter(data=torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.up_kernel = nn.Parameter(data=torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.down_kernel = nn.Parameter(data=torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.pool = nn.AvgPool2d(4)

    def forward(self, original, enhanced):
        # Calculate spatial loss between original and enhanced images
        # Modify this method based on your specific project requirements
        # You may want to compute the spatial loss here
        # and return the result
        pass

class ExposureLoss(nn.Module):

    def __init__(self, patch_size, mean_value):
        super(ExposureLoss, self).__init()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_value = mean_value

    def forward(self, x):
        # Calculate exposure loss based on mean value
        # Modify this method based on your specific project requirements
        # You may want to compute the exposure loss here
        # and return the result
        pass

class TotalVariationLoss(nn.Module):

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        # Calculate the total variation loss
        # Modify this method based on your specific project requirements
        # You may want to compute the total variation loss here
        # and return the result
        pass

class SaliencyLoss(nn.Module):

    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, x):
        # Calculate the saliency loss
        # Modify this method based on your specific project requirements
        # You may want to compute the saliency loss here
        # and return the result
        pass

class PerceptionLoss(nn.Module):

    def __init__(self):
        super(PerceptionLoss, self).__init__()

    def forward(self, x):
        # Calculate the perception loss
        # Modify this method based on your specific project requirements
        # You may want to compute the perception loss here
        # and return the result
        pass
