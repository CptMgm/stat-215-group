import yaml
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from autoencoder import ConvAutoencoder
from ..utils.data import CloudDataset, collate


# Load hyperparameter search configuration
with open("hyperparameter_search_config.yaml") as config_file:
    hyperparameter_config = yaml.safe_load(config_file)

# Set up data paths
train_images = ["../../data/image_data/image1.txt"]
validation_images = ["../../data/image_data/image2.txt"]


# Define the training function
def hyperparameter_search(config=None):
    with wandb.init(config=config):
        config = wandb.config
        config_id = wandb.run.id

        # Cross-validation setup
        print(f"Training model with config: {config}")

        # Load datasets
        train_dataset = CloudDataset(train_images, batch_size=config.batch_size)
        val_dataset = CloudDataset(validation_images, train=False)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate)

        # Initialize model
        model = ConvAutoencoder(
            embedding_size=config.embedding_size,
            loss_type=config.loss_type,
            kernel_size=config.kernel_size,
        )

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.patience,
            verbose=True
        )

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
            dirpath="checkpoints",
            filename=f"best-checkpoint-{config_id}"
        )

        # Set up trainer with early stopping, checkpointing, and logging
        trainer = Trainer(
            max_epochs=config.max_epochs,
            accelerator="mps",
            devices=1,
            logger=WandbLogger(),
            callbacks=[early_stopping, checkpoint_callback],
            log_every_n_steps=1,
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Load the best checkpoint
        best_model_path = checkpoint_callback.best_model_path
        print(f"Loading best model from {best_model_path}")
        best_model = ConvAutoencoder.load_from_checkpoint(best_model_path)

        # Validate with the best model
        results = trainer.validate(best_model, val_loader)
        val_loss = results[0]["val_loss"]

        # Log fold results to W&B
        wandb.log({"best_val_loss": val_loss})

if __name__ == "__main__":
    # Initialize sweep
    sweep_id = wandb.sweep(hyperparameter_config, project="autoencoder")

    # Start hyperparameter search
    print("Starting hyperparameter search...")
    wandb.agent(sweep_id, hyperparameter_search)
