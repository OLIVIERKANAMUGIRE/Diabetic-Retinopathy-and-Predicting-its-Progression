from Code0 import config, dataset, network, NODEs, train_node, small_NODEs
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint 
import os
import csv
#-------------------------------------------------

def train_latent_ode(autoencoder, 
                     train_loader, 
                     test_loader, 
                     latent_channels, 
                     solver, 
                     device='cuda', 
                     epochs=100, 
                     base_results_dir="/medip/experiments/okanamugire/SMALLNODERESULTS", 
                     run_id=1):

    # Construct path based on solver and latent dimension
    results_dir = os.path.join(base_results_dir, f"solver_{solver}", f"latent{latent_channels}")
    os.makedirs(results_dir, exist_ok=True)

    save_model_path = os.path.join(results_dir, f"best_latent_ode_run{run_id}.pth")
    csv_path = os.path.join(results_dir, f"losses_run{run_id}.csv")

    autoencoder.eval()

    # Define ODE model
    ode_func = NODEs.ConvLatentODEFuncAdaptive(latent_channels).to(device)
    ode_model = NODEs.LatentODEModel(ode_func, method=solver).to(device)

    optimizer = optim.Adam(ode_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    early_stop_patience = 10
    epochs_no_improve = 0

    time = torch.tensor([0., 1.]).to(device)

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        for epoch in tqdm(range(epochs)):
            ode_model.train()
            total_train_loss = 0.0

            for visit_1, visit_2 in train_loader:
                visit_1, visit_2 = visit_1.to(device), visit_2.to(device)

                with torch.no_grad():
                    _, z1 = autoencoder(visit_1)
                    _, z2 = autoencoder(visit_2)

                zt_pred = ode_model(z1, time)
                z2_pred = zt_pred[-1]

                loss = criterion(z2_pred, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            ode_model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for visit_1, visit_2 in test_loader:
                    visit_1, visit_2 = visit_1.to(device), visit_2.to(device)
                    _, z1 = autoencoder(visit_1)
                    _, z2 = autoencoder(visit_2)

                    zt_pred = ode_model(z1, time)
                    z2_pred = zt_pred[-1]

                    val_loss = criterion(z2_pred, z2)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(test_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
            csv_file.flush()

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(ode_model.state_dict(), save_model_path)
                print(f"Saved best model at epoch {epoch+1}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return ode_model

