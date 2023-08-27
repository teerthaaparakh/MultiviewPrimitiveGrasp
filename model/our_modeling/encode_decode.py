import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        modules = []
        in_dim = in_channels
        for h_dim in self.hidden_dims:
            modules.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.LeakyReLU()))
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dims)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dims)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, hidden_dims, in_dims, num_outputs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_dims = in_dims
        self.num_outputs = num_outputs
        modules = []
        self.decoder_input = nn.Linear(self.in_dims, self.hidden_dims[-1])
        input_dim = self.hidden_dims[-1]
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            h_dim = self.hidden_dims[i + 1]
            modules.append(nn.Sequential(nn.Linear(input_dim, h_dim), nn.LeakyReLU()))
            input_dim = h_dim
        self.decoder = nn.Sequential(*modules)
        self.final_layer_centerpoint = nn.Sequential(
            nn.Linear(input_dim, 2), nn.BatchNorm1d(2), nn.Sigmoid()
        )

        self.final_layer_offset = nn.Sequential(
            nn.Linear(input_dim, 8), nn.BatchNorm1d(8), nn.Tanh()
        )

    def forward(self, z):
        z = self.decoder_input(z)
        result = self.decoder(z)
        result_cp = self.final_layer_centerpoint(result)
        result_offset = self.final_layer_offset(result)
        return result_offset, result_cp
