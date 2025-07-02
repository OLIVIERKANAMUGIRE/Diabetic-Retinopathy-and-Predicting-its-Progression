import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from Code1 import config
#--------------------------------------------------------------------------------------------------

def train_autoencoder_cv(model_class, 
                         dataset, 
                         optimizer_fn, 
                         perceptual_loss_fn, 
                         device, 
                         latent_channels, 
                         epochs=100, 
                         patience=10, 
                         k_folds=5, 
                         base_output_dir="/medip/experiments/okanamugire"):

    # Output directory
    output_dir = os.path.join(base_output_dir, f"cv_results_{latent_channels}")
    os.makedirs(output_dir, exist_ok=True)

    # K-Fold Cross Validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold+1}/{k_folds} =====\n")
        
        fold_dir = os.path.join(output_dir, f"fold_{fold+1}")
        recon_dir = os.path.join(fold_dir, "reconstructions")
        latent_dir = os.path.join(fold_dir, "latent_space")
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)

        # Data Loaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=4, shuffle=False)

        # Model & optimizer instance for each fold
        model = model_class().to(device)
        optimizer = optimizer_fn(model.parameters())

        # Learning Rate Scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        best_val_loss = float('inf')
        early_stop_counter = 0
        save_path = os.path.join(fold_dir, "best_model.pth")

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # Training loop
            #******************************************************************************
            model.train()
            running_train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {epoch+1} [Training]"):
                x = batch.to(device)
                optimizer.zero_grad()
                x_recon, z = model(x)
                perceptual_loss = perceptual_loss_fn(x_recon, x)
                pixel_loss = F.mse_loss(x_recon, x)
                loss = pixel_loss + perceptual_loss
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

                del x, x_recon, z, loss, pixel_loss, perceptual_loss
                torch.cuda.empty_cache()

            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation loop
            #********************************************************************************
            
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Fold {fold+1} - Epoch {epoch+1} [Validation]"):
                    x = batch.to(device)
                    x_recon, _ = model(x)
                    perceptual_loss = perceptual_loss_fn(x_recon, x)
                    pixel_loss = F.mse_loss(x_recon, x)
                    loss = pixel_loss + perceptual_loss
                    running_val_loss += loss.item()

                    del x, x_recon, perceptual_loss, pixel_loss, loss
                    torch.cuda.empty_cache()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"Saved new best model at epoch {epoch+1}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Save reconstructions & latent space every 10 epochs
            if (epoch + 1) % 10 == 0:
                data_iter = iter(val_loader)
                val_images = next(data_iter).to(device)

                with torch.no_grad():
                    x_recon, z = model(val_images)

                np.save(os.path.join(latent_dir, f"latent_space_epoch_{epoch+1:03d}.npy"), z.cpu().numpy())

                x_original = val_images.cpu()
                x_reconstructed = x_recon.cpu()

                n = min(4, x_original.shape[0])
                fig, axs = plt.subplots(2, n, figsize=(12, 4))
                for i in range(n):
                    axs[0, i].imshow(x_original[i].squeeze().permute(1, 2, 0), cmap='gray')
                    axs[0, i].axis('off')
                    axs[0, i].set_title("Original")
                    axs[1, i].imshow(x_reconstructed[i].squeeze().permute(1, 2, 0), cmap='gray')
                    axs[1, i].axis('off')
                    axs[1, i].set_title("Reconstruction")

                plt.tight_layout()
                plt.savefig(os.path.join(recon_dir, f"epoch_{epoch+1:03d}.png"))
                plt.close()

                del val_images, x_recon, z, x_original, x_reconstructed
                torch.cuda.empty_cache()

        # Save per-fold loss history
        pd.DataFrame({
            "epoch": list(range(1, len(train_losses)+1)),
            "train_loss": train_losses,
            "val_loss": val_losses
        }).to_csv(os.path.join(fold_dir, "losses.csv"), index=False)

        # Store fold results
        fold_results.append({
            "fold": fold+1,
            "best_val_loss": best_val_loss
        })

    # Save cross-validation summary
    results_df = pd.DataFrame(fold_results)
    results_df['mean_val_loss'] = results_df['best_val_loss'].mean()
    results_df['std_val_loss'] = results_df['best_val_loss'].std()
    results_df.to_csv(os.path.join(output_dir, "cv_summary.csv"), index=False)

    print("\n Cross-Validation Complete!")
    print(results_df)
