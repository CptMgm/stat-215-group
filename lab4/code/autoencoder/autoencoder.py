import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(pl.LightningModule):
    """
    Convolutional autoencoder model for encoding Pixels into a lower-dimensional space.
    """
    def __init__(self, n_input_channels=8, embedding_size=8, loss_type="rmse", kernel_size=3):
        """
        Initialize the ConvAutoencoder model.

        :param n_input_channels: Number of input channels (default: 8).
        :param embedding_size: Size of the embedding space (default: 8).
        :param loss_type: Loss function type, either 'rmse' or 'mae' (default: 'rmse').
        :param kernel_size: Size of the convolutional kernel (default: 3).
        """
        super().__init__()
        self.n_input_channels = n_input_channels
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size

        # Validate and set the loss function type
        if loss_type not in ["rmse", "mae"]:
            raise ValueError("Invalid loss type. Choose either 'rmse' or 'mae'.")
        self.loss_type = loss_type

        # Save hyperparameters
        self.save_hyperparameters()

        # Use same padding to ensure the output shape matches the input shape
        padding = "same"

        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, embedding_size, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(),
        )

        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_size, 256, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, n_input_channels, kernel_size=self.kernel_size, padding=padding),
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder model.

        :param x: Input tensor of shape (batch_size, n_input_channels, height, width).
        :return: Decoded output tensor of shape (batch_size, n_input_channels, height, width).
        """
        # Pass through the encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Ensure the output shape matches the input shape
        assert decoded.shape == x.shape, (
            f"Output shape {decoded.shape} does not match input shape {x.shape}. "
            "Ensure the kernel size and padding are configured correctly."
        )

        return decoded

    def _compute_loss(self, outputs, inputs):
        if self.loss_type == "rmse":
            return torch.sqrt(F.mse_loss(outputs, inputs))
        elif self.loss_type == "mae":
            return F.l1_loss(outputs, inputs)

    def training_step(self, batch, batch_idx):
        """
        Perform a training step on a batch of data.

        :param batch: Batch of data containing inputs and labels.
        :param batch_idx: Index of the current batch.
        :return: Loss value for the current batch.
        """
        inputs, _ = batch
        outputs = self(inputs)
        loss = self._compute_loss(outputs, inputs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step on a batch of data.

        :param batch: Batch of data containing inputs and labels.
        :param batch_idx: Index of the current batch.
        :return: Loss value for the current batch.
        """
        inputs, _ = batch
        outputs = self(inputs)
        loss = self._compute_loss(outputs, inputs)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training the model.

        :return: Optimizer to use for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def embed(self, x):
        """
        Encode the input tensor into the embedding space.

        :param x: Input tensor of shape (batch_size, n_input_channels, height, width).
        :return: Encoded tensor of shape (batch_size, embedding_size, height, width).
        """
        return self.encoder(x)
