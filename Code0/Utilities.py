import torch
import torch.nn as nn
from geomloss import SamplesLoss
from Code0 import config
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = vgg[:16].eval()  

        for param in self.vgg_layers.parameters():
            param.requires_grad = False  

        self.layer_weights = layer_weights if layer_weights else [1.0]

        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        # Normalize images to VGG's expected range
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2

        loss = 0.0
        for idx, weight in enumerate(self.layer_weights):
            x = self.vgg_layers[idx](x)
            y = self.vgg_layers[idx](y)
            loss += weight * self.criterion(x, y)
        return loss

