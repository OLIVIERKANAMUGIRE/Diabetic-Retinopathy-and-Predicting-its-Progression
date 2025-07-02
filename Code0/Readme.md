### dataset.py
------------------------------------------------------
    This file provides custom classdata that provides data with index as well as preprocessing function.

        Input: Path to the root dataset directory.
        Output: For each patient, returns a tuple of two preprocessed and transformed images: (visit1_image, visit2_image).

    Preprocessing Details:
        Images are loaded using OpenCV.
        Converted from BGR to RGB.
        Resized to a target resolution which is 512x512.

    Enhanced using CLAHE applied to each channel independently.
    Normalized to the [0, 1] range.

    Dependencies
        - torch
        - torchvision
        - PIL
        - cv2 (OpenCV)
        - numpy
        - matplotlib 


### config.py
---------------------------------------------------------
    This file contains the set of hyperparameters that was used to train autoencoder model as well as the Neural ordinary differential equation.

    Example:
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

### NODEs.py
--------------------------------------------------------
    This file contains classes for the Neural Ordinary Differential Equations using concvolutional neural nets.

    1. ConvLatentODEFunc
        This class defines the dynamics function of the ODE using convolutional layers and Group Normalization.

        Input: Tensor z representing latent image features.
        Output: Derivative dz/dt to be used by the ODE solver.

    2. LatentODEModel
        This wraps the dynamics in a PyTorch module, using torchdiffeq.odeint to integrate over time.

        z0: Initial latent state (e.g., output of an encoder).
        t: A sequence of time points (e.g., [0, 1]).

### netwok.py
-------------------------------------------------------------
    THis is former autoencoder model using in the first experiments.
    

### pretrain_autoencoders.py
-------------------------------------------------------------
    