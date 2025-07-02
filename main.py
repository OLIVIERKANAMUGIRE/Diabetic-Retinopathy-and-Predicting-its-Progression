from Code1 import Utilities, config, network, train_and_test_loop, dataset

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torch.utils.data import ConcatDataset


import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#####################################################################

# LOADING DATASET
#---------------------------------------------------
dataset1 = dataset.RetinalDataset("retinal_images_uku_20070131", target_size= config.TARGET_SIZE)
dataset2 = dataset.RetinalDataset("/medip/experiments/okanamugire/B. Disease Grading/1. Original Images/a. Training Set", target_size= config.TARGET_SIZE)

full_dataset = ConcatDataset([dataset1, dataset2])

# Set random seed for reproducibility
seed = config.SEED  
torch.manual_seed(seed)  
np.random.seed(seed)  
random.seed(seed)  

# Ensure reproducibility on CUDA devices 
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  

# For deterministic operations 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataset split
train_size = int(config.TRAIN_RATIO * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

#####################################################################################
# LOAD THE PERCEPTUAL LOSS FUNCTIONS FROM UTILS
perceptual_loss_fn = Utilities.PerceptualLoss()
perceptual_loss_fn = perceptual_loss_fn.to(config.DEVICE)

####################################

list_of_latents = config.LATENT_CHANNELS

if __name__ == "__main__":
    for latent_channel in tqdm(list_of_latents):
        # INITIALIZE THE MODEL and OPTIMIZER AND PRINT THE SUMMARY
        model = network.Autoencoder(input_channels=3, latent_channels= latent_channel).to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        print("MODEL SUMMARY FOR {latent_channel}")
        print(summary(model, input_size=(4, 3, 512, 512), depth=4, col_names=["input_size", "output_size", "num_params"]))
        train_and_test_loop.train_autoencoder(model, 
                      train_loader, test_loader, 
                      optimizer, 
                      perceptual_loss_fn, 
                      device = config.DEVICE, 
                      latent_channels=latent_channel, 
                      epochs=100, 
                      patience=10, 
                      base_output_dir="/medip/experiments/okanamugire")



