from Code1 import Utilities, config, network, cross_validation, dataset

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset
from torchinfo import summary

import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

os.environ["CUDA_LAUNCH_BLOCKING"] = "2"


#####################################################################

# LOADING DATASET

dataset1 = dataset.RetinalDataset("retinal_images_uku_20070131", target_size=config.TARGET_SIZE)
dataset2 = dataset.RetinalDataset("/medip/experiments/okanamugire/B. Disease Grading/1. Original Images/a. Training Set", target_size=config.TARGET_SIZE)
dataset3 = dataset.RetinalDataset("/medip/experiments/okanamugire/aptos2019-blindness-detection/train_images", target_size=config.TARGET_SIZE)
full_dataset = ConcatDataset([dataset1, dataset2, dataset3])


# LOAD PERCEPTUAL LOSS FUNCTION
perceptual_loss_fn = Utilities.PerceptualLoss().to(config.DEVICE)

#################################### train_autoencoder_cv

list_of_latents = config.LATENT_CHANNELS

if __name__ == "__main__":
    for latent_channel in tqdm(list_of_latents):
        print(f"MODEL SUMMARY FOR latent_channel = {latent_channel}")

        # For model summary only (no gradients)
        temp_model = network.Autoencoder(input_channels=3, latent_channels=latent_channel).to(config.DEVICE)
        with torch.no_grad():
            print(summary(temp_model, input_size=(4, 3, 512, 512), depth=4, col_names=["input_size", "output_size", "num_params"]))
        
        # Free GPU memory after summary
        del temp_model
        torch.cuda.empty_cache()

        # Define optimizer factory function 
        optimizer_fn = lambda params: torch.optim.Adam(params, lr=config.LR)

        # Train with CV 
        cross_validation.train_autoencoder_cv(
                model_class=lambda: network.Autoencoder(input_channels=3, latent_channels=latent_channel),
                dataset=full_dataset,
                optimizer_fn=optimizer_fn,
                perceptual_loss_fn=perceptual_loss_fn,
                device=config.DEVICE,
                latent_channels=latent_channel,
                epochs=100,
                patience=10,
                base_output_dir="/medip/experiments/okanamugire",
                k_folds=5
            )
