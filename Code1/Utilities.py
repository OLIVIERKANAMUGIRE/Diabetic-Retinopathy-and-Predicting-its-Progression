import torch
import torch.nn as nn
from geomloss import SamplesLoss
from Code1 import config
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



# KULLBACK LEIBLER DIVERGENCE (LATENT DISTRIBUTION AND NORMAL DISTRIBUTION OR BETA DISTRIBUTION)

def vae_gaussian_kl_loss(mu, logvar):
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    return KLD.mean()

def reconstruction_loss(x_reconstructed, x, loss_name):
    if loss_name  ==  "Mean Squared Error":
        loss_value = nn.MSELoss()
    elif loss_name == "Cross Entropy":
        loss_value = nn.BCELoss()
    return loss_value

def vae_loss(y_pred, y_true):
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    return config.ALPHA*recon_loss + kld_loss

def latent_loss(h_pred,h_true):
    sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05) 
    loss = sinkhorn_loss_fn(h_pred, h_true)
    return loss
