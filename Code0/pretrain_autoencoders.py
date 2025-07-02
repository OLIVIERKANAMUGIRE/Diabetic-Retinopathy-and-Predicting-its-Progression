from Code0 import network
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm

#------------------------------------------------------------------------------------
def train_autoencoder(model, 
                      train_loader, test_loader, 
                      optimizer, 
                      perceptual_loss_fn, 
                      device, 
                      latent_channels, 
                      epochs=100, 
                      patience=10):

    base_dir = "/medip/experiments/okanamugire/pretrained_autoencoder_results_64"
    recon_dir = os.path.join(base_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)

    model_path = os.path.join(base_dir, "best_pretrained_autoencoder_model.pth")
    csv_path = os.path.join(base_dir, "training_logs.csv")

    train_losses = []
    test_losses = []

    best_loss = float('inf')
    early_stop_counter = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_train_loss = 0.0

        for visit_1, visit_2 in train_loader:
            visit_1 = visit_1.to(device)
            visit_2 = visit_2.to(device)

            optimizer.zero_grad()

            recon1, _ = model(visit_1)
            recon2, _ = model(visit_2)

            perceptual_loss = perceptual_loss_fn(recon1, visit_1) + perceptual_loss_fn(recon2, visit_2)
            pixel_loss = F.mse_loss(recon1, visit_1) + F.mse_loss(recon2, visit_2)
            loss = pixel_loss + perceptual_loss

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for visit_1, visit_2 in test_loader:
                visit_1 = visit_1.to(device)
                visit_2 = visit_2.to(device)

                recon1, _ = model(visit_1)
                recon2, _ = model(visit_2)

                perceptual_loss = perceptual_loss_fn(recon1, visit_1) + perceptual_loss_fn(recon2, visit_2)
                pixel_loss = F.mse_loss(recon1, visit_1) + F.mse_loss(recon2, visit_2)
                loss = pixel_loss + perceptual_loss

                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            early_stop_counter = 0
            print(f"New best model at epoch {epoch+1}")

            # Save best model
            torch.save(model.state_dict(), model_path)

            # Save reconstructions
            visit_1, visit_2 = next(iter(test_loader))
            visit_1 = visit_1.to(device)
            visit_2 = visit_2.to(device)
            with torch.no_grad():
                recon1, _ = model(visit_1)
                recon2, _ = model(visit_2)

            x_original1 = visit_1.cpu()
            x_recon1 = recon1.cpu()
            x_original2 = visit_2.cpu()
            x_recon2 = recon2.cpu()

            n = min(4, x_original1.size(0))
            fig, axs = plt.subplots(4, n, figsize=(12, 6))
            for i in range(n):
                axs[0, i].imshow(x_original1[i].permute(1, 2, 0).clamp(0, 1))
                axs[0, i].axis('off')
                axs[0, i].set_title("Visit 1")

                axs[1, i].imshow(x_recon1[i].permute(1, 2, 0).clamp(0, 1))
                axs[1, i].axis('off')
                axs[1, i].set_title("Recon 1")

                axs[2, i].imshow(x_original2[i].permute(1, 2, 0).clamp(0, 1))
                axs[2, i].axis('off')
                axs[2, i].set_title("Visit 2")

                axs[3, i].imshow(x_recon2[i].permute(1, 2, 0).clamp(0, 1))
                axs[3, i].axis('off')
                axs[3, i].set_title("Recon 2")

            plt.tight_layout()
            recon_path = os.path.join(recon_dir, f"best_recon_epoch_{epoch+1}.png")
            plt.savefig(recon_path)
            plt.close()

        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save training loss to CSV
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Test Loss"])
        for i, (tr, te) in enumerate(zip(train_losses, test_losses), 1):
            writer.writerow([i, tr, te])
