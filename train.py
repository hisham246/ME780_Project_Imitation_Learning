import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb

# Import model and dataset classes
from model_il import ImitationNet
from dataset import ImitationDataset

# Initialize paths
data_dir = '/home/hisham246/uwaterloo/robohub/imitation_learning_tb4/data/diffusion'
model_dir = '/home/hisham246/uwaterloo/robohub/imitation_learning_tb4/models/lstm'

def train_model(model, data_dir, model_dir, epochs=50, batch_size=50, lr=0.002, lr_decay=0.1, device='cuda'):
    # Initialize wandb
    wandb.init(
        project="turtlebot_diffusion",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "lr_decay": lr_decay,
            "device": device.type,
            "mode": "lstm"
        },
    )
    config = wandb.config

    # Load data
    train_data = ImitationDataset(device=device, mode='lstm')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Set up the optimizer and loss tracking
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr_decay)
    model.train()
    losses = []

    # Training Loop
    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, (pose, scan, vel) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.step(odom_input=pose, laser_scan=scan, target_control=vel)
            loss.backward()
            optimizer.step()

            # Log batch loss
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            wandb.log({"batch_loss": batch_loss, "epoch": epoch + 1})

        # Log average epoch loss
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        # Save model checkpoints and upload to wandb
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save({'model': model.state_dict()}, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
            # Upload checkpoint to wandb
            wandb.save(checkpoint_path)

    # Final model save
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_final.pt")
    torch.save({'model': model.state_dict()}, model_path)
    print(f"Final model saved at {model_path}")
    wandb.save(model_path)

# Main function for argument parsing and training
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train imitation learning model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training (cpu or cuda)')
    args = parser.parse_args()

    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = ImitationNet(control_dim=2, device=device).to(device)

    # Train the model
    train_model(model, data_dir, model_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lr_decay=args.lr_decay, device=device)
