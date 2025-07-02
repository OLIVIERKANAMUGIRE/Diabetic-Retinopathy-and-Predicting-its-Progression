# This config file id for training VAE
#------------------------------------------------

import torch

DEVICE  = (
    'cuda:1'
    if torch.cuda.is_available()
    # else 'mps'
    # if torch.backends.mps.is_available ()
    else 'cpu'
)

# Model hyperparametes
#--------------------------------------------------------------------

LR = 1e-4
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE_PATIENCE = 3

IMAGE_SIZE = 1200 
CHANNELS = 3 

BATCH_SIZE  = 4
EMBEDING_DIM = 100
EPOCHS =100
SEED = 1 
TRAIN_RATIO = 0.8

LATENT_CHANNELS = [2,4,8,16,32,64]
TARGET_SIZE = (512, 512)
NUMBER_OF_IMAGES_TO_DISPLAY = 3 # not greater than 4
