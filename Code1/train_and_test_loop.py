
import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_autoencoder(model, 
                      train_loader, test_loader, 
                      optimizer, 
                      perceptual_loss_fn, 
                      device, 
                      latent_channels, 
                      epochs=100, 
                      patience=10, 
                      base_output_dir="/medip/experiments"):
    
    # Create output directory based on latent_channels
    output_dir = os.path.join(base_output_dir, f"output_results_{latent_channels}_v2")
    recon_dir = os.path.join(output_dir, "reconstructions")
    latent_dir = os.path.join(output_dir, "latent_space")  
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)

    # Store losses
    train_losses = []
    test_losses = []

    # Early Stopping and LR scheduler settings
    best_loss = float('inf')
    early_stop_counter = 0
    save_path = os.path.join(output_dir, "best_model.pth")

    # LR scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()

            x_recon, z = model(x)
            perceptual_loss = perceptual_loss_fn(x_recon, x)
            pixel_loss = F.mse_loss(x_recon, x)
            loss = pixel_loss + perceptual_loss

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(device)
                x_recon, _ = model(x)

                perceptual_loss = perceptual_loss_fn(x_recon, x)
                pixel_loss = F.mse_loss(x_recon, x)
                loss = pixel_loss + perceptual_loss

                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model at epoch {epoch+1}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % 10 == 0:
            data_iter = iter(test_loader)
            test_images = next(data_iter).to(device)
            with torch.no_grad():
                x_recon, z = model(test_images)

            latent_file = os.path.join(latent_dir, f"latent_space_epoch_{epoch+1:03d}.npy")
            np.save(latent_file, z.cpu().numpy())

            x_original = test_images.cpu()
            x_reconstructed = x_recon.cpu()

            n = 4
            fig, axs = plt.subplots(2, n, figsize=(12, 4))
            for i in range(n):
                axs[0, i].imshow(x_original[i].squeeze().permute(1, 2, 0), cmap='gray')
                axs[0, i].axis('off')
                axs[0, i].set_title("Original")
                axs[1, i].imshow(x_reconstructed[i].squeeze().permute(1, 2, 0), cmap='gray')
                axs[1, i].axis('off')
                axs[1, i].set_title("Reconstruction")

            plt.tight_layout()
            recon_path = os.path.join(recon_dir, f"epoch_{epoch+1:03d}.png")
            plt.savefig(recon_path)
            plt.close()

    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "test_loss": test_losses
    })
    loss_df.to_csv(os.path.join(output_dir, "losses.csv"), index=False)
