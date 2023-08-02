import numpy as np 
import torch 
from torch import nn

class Encoder(nn.Module):
    def __init__(self, hidden_dims, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                    nn.LazyLinear(h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())

        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*4, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*4, self.latent_dim)
            
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]
    

        
class Decoder(nn.Module):
    def __init__(self, hidden_dims, latent_dims, num_outputs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims
        self.num_outputs = num_outputs
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * 4)
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            h_dim = self.hidden_dims[i+1]
            modules.append(
                nn.Sequential(
                    nn.Linear(h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.LazyLinear(self.num_outputs),
                            nn.LeakyReLU())
                            
        
        
    def forward(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
        
        
        