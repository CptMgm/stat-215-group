import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import gc
from tqdm import tqdm
from patchdataset import PatchDataset
from data import make_data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from autoencoder import Autoencoder

# ... existing imports ...

# Define configuration directly in the script
config = {
    "data": {
        "patch_size": 9  # Example value, adjust as needed
    },
    "optimizer": {
        # Add optimizer configuration here
    },
    "checkpoint": {
        "dirpath": "checkpoints",  # Directory to save checkpoints
        "filename": "best-checkpoint",  # Checkpoint filename
        "save_top_k": 1,  # Save only the best model
        "monitor": "val_loss",  # Metric to monitor
        "mode": "min"  # Minimize the monitored metric
    },
    "wandb": {
        # Add wandb configuration here
    },
    "trainer": {
        # Add trainer configuration here
    }
}

# Clean up memory
gc.collect()
torch.cuda.empty_cache()

# Make the patch data
_, patches = make_data(patch_size=config["data"]["patch_size"])
all_patches = patches[0] + patches[1] + patches[2]

# ... rest of the existing code ...

# Configure the settings for making checkpoints
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

# ... rest of the existing code ...

# Load and Prepare Data
# Removed YAML config loading and assertions

# Clean up memory
gc.collect()
torch.cuda.empty_cache()

# Make the patch data
_, patches = make_data(patch_size=config["data"]["patch_size"])
all_patches = patches[0] + patches[1] + patches[2]

# Randomly do train/val split by individual patches
train_bool = np.random.rand(len(all_patches)) < 0.8
train_idx = np.where(train_bool)[0]
val_idx = np.where(~train_bool)[0]

# Create train and val datasets
train_patches = [all_patches[i] for i in train_idx]
val_patches = [all_patches[i] for i in val_idx]
train_dataset = PatchDataset(train_patches)
val_dataset = PatchDataset(val_patches)

# Create train and val dataloaders
dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the Autoencoder Model
model = Autoencoder(
    optimizer_config=config["optimizer"],
    patch_size=config["data"]["patch_size"],
)

# Configure the settings for making checkpoints
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

# Initialize the wandb logger, giving it our config file to save, and also configuring the logger itself.
wandb_logger = WandbLogger(config=config, **config["wandb"])

# Initialize the trainer
trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=50, log_every_n_steps=10, **config["trainer"]
)

# Train the Autoencoder
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

# Embedding Evaluation
def plot_embeddings(data_loader, autoencoder, title):
    autoencoder.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            encoded = autoencoder.embed(batch)
            embeddings.append(encoded.numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    # Plot Embeddings
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], palette="coolwarm")
    plt.title(title)
    plt.show()

plot_embeddings(dataloader_train, model, "Train Set Embeddings")
plot_embeddings(dataloader_val, model, "Validation Set Embeddings")

# Train a New Autoencoder with a Different Architecture
class ImprovedAutoencoder(Autoencoder):
    def __init__(self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8):
        super().__init__(optimizer_config, n_input_channels, patch_size, embedding_size)
        input_size = int(n_input_channels * (patch_size**2))
        
        # Modify Encoder and Decoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )

# Train the Improved Autoencoder
improved_autoencoder = ImprovedAutoencoder(optimizer_config=config["optimizer"])
trainer.fit(improved_autoencoder, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

# Plot Improved Embeddings
plot_embeddings(dataloader_train, improved_autoencoder, "Improved Train Set Embeddings")
plot_embeddings(dataloader_val, improved_autoencoder, "Improved Validation Set Embeddings")
