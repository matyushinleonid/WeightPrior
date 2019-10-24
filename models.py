from __future__ import print_function
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist
from torchvision import datasets, transforms
import torch.utils.data as utils

import numpy as np
from tqdm import tqdm_notebook as tqdm

from matplotlib import pyplot as plt

class VAE7x7(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super(VAE7x7, self).__init__()

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim * 2, z_dim * 2, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim * 2, 3),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim, 2, 1)
        )

    def encode(self, input):
        mean_and_logvar = self.encoder(input)
        [z_mean, z_logvar] = [mean_and_logvar[:,:self.z_dim,:,:],
                              mean_and_logvar[:,self.z_dim:,:,:]]
        
        return z_mean, z_logvar

    def decode(self, input):
        mean_and_logvar = self.decoder(input)
        [x_mean, x_logvar] = [mean_and_logvar[:,:1,:,:],
                              mean_and_logvar[:,1:,:,:]]
        
        return x_mean, x_logvar

    def forward(self, input):
        z_mean, z_logvar = self.encode(input)
        z = dist.Normal(z_mean, z_logvar.mul(0.5).exp_()).rsample()
        x_mean, x_logvar = self.decode(z)

        return z_mean, z_logvar, z, x_mean, x_logvar
    
    def elbo(self, x, beta=1.):
        z_mean, z_logvar, z, x_mean, x_logvar = self.forward(x)
        
        loglikelihood = 1/2 * (x_logvar + (x - x_mean).pow(2) / x_logvar.exp())
        loglikelihood = loglikelihood.sum()
        
        kl = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl = kl.sum()
        
        #prior = dist.Normal(torch.FloatTensor([0.]).cuda(), torch.FloatTensor([1.]).cuda())
        #loglikelihood = -dist.Normal(x_mean, x_logvar.mul(0.5).exp_()).log_prob(x).sum()
        #kl = dist.kl_divergence(dist.Normal(z_mean, z_logvar.mul(0.5).exp_()), prior).sum()
        
        return loglikelihood + beta * kl
    
    def generate(self, n=1):
        z = torch.randn(n, self.z_dim, 1, 1).cuda()
        x_mean, _ = self.decode(z)
        
        return x_mean
    
    
    
    
class VAE5x5(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super(VAE5x5, self).__init__()

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim * 2, z_dim * 2, 3)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim * 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.ELU(),
            nn.Conv2d(hidden_dim, 2, 1)
        )

    def encode(self, input):
        mean_and_logvar = self.encoder(input)
        [z_mean, z_logvar] = [mean_and_logvar[:,:self.z_dim,:,:],
                              mean_and_logvar[:,self.z_dim:,:,:]]
        
        return z_mean, z_logvar

    def decode(self, input):
        mean_and_logvar = self.decoder(input)
        [x_mean, x_logvar] = [mean_and_logvar[:,:1,:,:],
                              mean_and_logvar[:,1:,:,:]]
        [x_mean, x_logvar] = [torch.tanh(x_mean), 10 * torch.tanh(x_logvar)]
        
        return x_mean, x_logvar

    def forward(self, input):
        z_mean, z_logvar = self.encode(input)
        z = dist.Normal(z_mean, z_logvar.mul(0.5).exp_()).rsample()
        x_mean, x_logvar = self.decode(z)

        return z_mean, z_logvar, z, x_mean, x_logvar
    
    def elbo(self, x, beta=1.):
        z_mean, z_logvar, z, x_mean, x_logvar = self.forward(x)
        
        loglikelihood = 1/2 * (x_logvar + (x - x_mean).pow(2) / x_logvar.exp())
        loglikelihood = loglikelihood.sum()
        
        kl = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl = kl.sum()
        
        #prior = dist.Normal(torch.FloatTensor([0.]).cuda(), torch.FloatTensor([1.]).cuda())
        #loglikelihood = -dist.Normal(x_mean, x_logvar.mul(0.5).exp_()).log_prob(x).sum()
        #kl = dist.kl_divergence(dist.Normal(z_mean, z_logvar.mul(0.5).exp_()), prior).sum()
        
        return loglikelihood + beta * kl
    
    def generate(self, n=1):
        z = torch.randn(n, self.z_dim, 1, 1).cuda()
        x_mean, _ = self.decode(z)
        
        return x_mean