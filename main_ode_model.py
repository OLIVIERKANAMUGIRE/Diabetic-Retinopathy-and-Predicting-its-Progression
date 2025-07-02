# from Code0 import config, dataset, network, NODEs, train_node, pretrain_autoencoders, Utilities
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# from PIL import Image
# import cv2

# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms

# import matplotlib.pyplot as plt
# import numpy as np
# ############################################################################
# import gc

if __name__ == "__main__":
    import gc
    from Code0 import dataset, network, train_node, Utilities, config
    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    import random
    import numpy as np

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)  

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    image_folder = "/medip/experiments/okanamugire/LongitudinalDRdataset/Oculus Dexter"
    dataset = dataset.LongitudinalFundusDataset(image_folder, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    latent_channels_list = config.LATENT_CHANNELS
    solvers = config.ADAP_STEP_SOLVERS

    for latent_channels in latent_channels_list:
        model = network.Autoencoder(input_channels=3, latent_channels=latent_channels).to(device)

        # Load corresponding pretrained weights
        pretrained_path = f"/medip/experiments/okanamugire/pretrained_autoencoder_results_{latent_channels}/best_pretrained_autoencoder_model.pth"
        print(pretrained_path)
        if not os.path.exists(pretrained_path):
            print(f"Missing pretrained AE: {pretrained_path}")
            continue

        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.eval()

        for solver in solvers:
            print(f"\n***** Training: latent={latent_channels}, solver={solver} *****\n")

            gc.collect()
            torch.cuda.empty_cache()

            ode_model = train_node.train_latent_ode(
                autoencoder=model,
                train_loader=train_loader,
                test_loader=test_loader,
                latent_channels=latent_channels,
                solver=solver,
                device=device,
                epochs=100,
                base_results_dir="/medip/experiments/okanamugire/SMALLNODERESULTS", 
                run_id="1"
            )

