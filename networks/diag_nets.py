
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from networks import wide_resnet
import copy
import numpy as np
from numpy.random import beta

def Featurizer(input_shape, hparams, args):
    """Auto-select an appropriate featurizer for the given input shape."""
    if args.nets_base in ['diag_nets','LeNet']:
        return LeNet_1d_Featurizer(input_shape[0])#LeNet_1d_Featurizer() # hparams can be added if needed
    elif args.nets_base in ['Transformer']:
        return Transformer_1d_Featurizer(input_dim=input_shape[0], seq_length=input_shape[1])
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

# -----------------------input size>=32---------------------------------
class LeNet_1d_Featurizer(nn.Module):
    def __init__(self, in_channel=1):
        super(LeNet_1d_Featurizer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, 6, 5),
            nn.BatchNorm1d(6),  # 64
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 25, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.n_outputs =84

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Transformer_1d_Featurizer(nn.Module):
    def __init__(self, input_dim,  seq_length, d_model=128, nhead=2, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        """
        Args:
            input_dim (int): Number of input features per time step in the signal, like channel_num.
            num_classes (int): Number of output classes for classification.
            seq_length (int): Length of the input sequence.
            d_model (int): Dimension of the model (embedding dimension).
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(Transformer_1d_Featurizer, self).__init__()
        self.n_outputs = d_model
        # Featurizer: Input embedding and Transformer Encoder
        self.input_embedding = nn.Linear(input_dim, d_model)  # Project input features to d_model
        self.positional_encoding = self._generate_positional_encoding(seq_length, d_model)  # Positional encoding

        # # Use batch_first=True for better inference performance
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     batch_first=True  # Set batch_first to True
        # )

        # Use batch_first=True for better inference performance
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def _generate_positional_encoding(self, seq_length, d_model):
        """
        Generate positional encoding for the input sequence.
        """
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim, seq_length).

        Returns:
            Tensor: Logits of shape (batch_size, num_classes).
        """
        # (batch_size, seq_length, input_dim) for embedding
        x = x.permute(0, 2, 1)

        batch_size, seq_length, input_dim = x.shape

        # Input embedding
        x = self.input_embedding(x)  # Shape: (batch_size, seq_length, d_model)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_length, :].to(x.device)

        # Transformer Encoder (for pytorch at least 1.9.0, batch_first=True, so no need to permute dimensions)
        # For lower version, Transformer expects input as (seq_length, batch_size, d_model), so permute
        x = x.permute(1, 0, 2)  # Shape: (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)  # Shape: (seq_length, batch_size, d_model)
        # Permute back to (batch_size, seq_length, d_model)
        x = x.permute(1, 0, 2)  # Shape: (batch_size, seq_length, d_model)
        # Take the mean of the sequence (global average pooling over the sequence dimension)
        x = x.mean(dim=1)  # Shape: (batch_size, d_model)

        return x
