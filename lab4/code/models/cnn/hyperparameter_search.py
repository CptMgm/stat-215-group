import yaml
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from model import UNetSegmenter
from ...utils.data import CloudDataset, collate

# Load hyperparameter search configuration
with open("hyperparameter_search_config.yaml") as config_file:
    hyperparameter_config = yaml.safe_load(config_file)

# Set up data paths
train_images = ["../../../data/image_data/image1.txt"]
validation_images = ["../../../data/image_data/image2.txt"]


# Define the training function
def hyperparameter_search(config=None):
    with wandb.init(config=config):
        config = wandb.config
        config_id = wandb.run.id

        # Cross-validation setup
        print(f"Training model with config: {config}")

        # Load datasets
        train_dataset = CloudDataset(train_images, augment=config.image_augmentation, batch_size=config.batch_size)
        val_dataset = CloudDataset(validation_images, augment=False, train=False)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate)

        # Initialize model
        model = UNetSegmenter()

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.patience,
            verbose=True
        )

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
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
            callbacks=[early_stopping, checkpoint_callback]
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Load the best checkpoint
        best_model_path = checkpoint_callback.best_model_path
        print(f"Loading best model from {best_model_path}")
        best_model = UNetSegmenter.load_from_checkpoint(best_model_path)

        # Validate with the best model
        results = trainer.validate(best_model, val_loader)
        accuracy = results[0]["val_accuracy"]

        # Log fold results to W&B
        wandb.log({"best_val_accuracy": accuracy})

if __name__ == "__main__":
    # Initialize sweep
    sweep_id = wandb.sweep(hyperparameter_config, project="cloud_detection")

    # Start hyperparameter search
    print("Starting hyperparameter search...")
    wandb.agent(sweep_id, hyperparameter_search)
