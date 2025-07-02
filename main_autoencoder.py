from Code0 import config, dataset, network, NODEs, train_node, pretrain_autoencoders, Utilities
import os
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
############################################################################

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

image_folder = "/medip/experiments/okanamugire/LongitudinalDRdataset/Oculus Dexter"
dataset = dataset.LongitudinalFundusDataset(image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size =4, shuffle=True)

train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
torch.manual_seed(1)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


### load a pretrained model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = network.Autoencoder(input_channels=3, latent_channels=64).to(device)
model.load_state_dict(torch.load("/medip/experiments/okanamugire/output_results_64/best_model.pth", 
                                 map_location=device))
print("Loaded pretrained weights from best_model.pth.")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#-------------------------------------------------------------------------------------------------

if __name__== "__main__":
    # LOAD THE PERCEPTUAL LOSS FUNCTIONS FROM UTILS
    perceptual_loss_fn = Utilities.PerceptualLoss()
    perceptual_loss_fn = perceptual_loss_fn.to(device=device)

    pretrain_autoencoders.train_autoencoder(model, 
                        train_loader, test_loader, 
                        optimizer, 
                        perceptual_loss_fn, 
                        device, 
                        latent_channels = 64, 
                        epochs=100, 
                        patience=10)
##------------------------------------------------------------------------------------------------


