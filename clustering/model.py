from torch import nn
import torch
# Need to set the auto encoder properly.

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        # Encoder: (B, 3, 512, 128) -> (B, 32, 32, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   # -> (B, 16, 512, 128)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # -> (B, 16, 256, 64)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> (B, 32, 256, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # -> (B, 32, 128, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # -> (B, 64, 128, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # -> (B, 64, 64, 16)

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # -> (B, 32, 64, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)                               # -> (B, 32, 32, 8)
        )

        self.flatten = nn.Flatten()
        self.to_bottleneck = nn.Linear(32 * 32 * 8, latent_dim)     # -> (B, 512)
        self.from_bottleneck = nn.Linear(latent_dim, 32 * 32 * 8)    # -> (B, 8192)

        # Decoder: (B, 32, 32, 8) -> (B, 3, 512, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),    # -> (B, 64, 64, 16)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # -> (B, 32, 128, 32)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # -> (B, 16, 256, 64)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),     # -> (B, 3, 512, 128)
            nn.Sigmoid()  # use if inputs are normalized to [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)                         # (B, 32, 32, 8)
        x = self.flatten(x)                         # (B, 8192)
        z = self.to_bottleneck(x)                   # (B, 512)
        return z

    def decode(self, z):
        x = self.from_bottleneck(z)                 # (B, 8192)
        x = x.view(-1, 32, 32, 8)                   # (B, 32, 32, 8)
        x = self.decoder(x)                         # (B, 3, 512, 128)
        return x

    def forward(self, x):
        z = self.encode(x)                          # bottleneck
        recon = self.decode(z)                      # reconstruction
        return recon


class Varational_AE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        # Encoder: (B, 3, 512, 128) -> (B, 32, 32, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   # -> (B, 16, 512, 128)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # -> (B, 16, 256, 64)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> (B, 32, 256, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # -> (B, 32, 128, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # -> (B, 64, 128, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # -> (B, 64, 64, 16)

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # -> (B, 32, 64, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)                               # -> (B, 32, 32, 8)
        )

        self.flatten = nn.Flatten()
        
        # Intermediate compression step before splitting
        self.to_bottleneck = nn.Linear(32 * 32 * 8, 512)     # -> (B, 512)

        # Mean and variance layers
        self.mu = nn.Linear(512, latent_dim)
        self.log_var = nn.Linear(512, latent_dim)

        # FIX 2: Changed input size from 512 to latent_dim (256)
        self.from_bottleneck = nn.Linear(latent_dim, 32 * 32 * 8)    # -> (B, 8192)

        # Decoder: (B, 32, 32, 8) -> (B, 3, 512, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),    # -> (B, 64, 64, 16)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # -> (B, 32, 128, 32)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # -> (B, 16, 256, 64)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),     # -> (B, 3, 512, 128)
            nn.Sigmoid()  
        )

    def encode(self, x):
        x = self.encoder(x)                         
        h1 = self.flatten(x)                         
        h2 = self.to_bottleneck(h1) # Pass through intermediate layer
        return self.mu(h2), self.log_var(h2)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            # FIX 1: Changed rand_like to randn_like for normal distribution
            eps = torch.randn_like(std) 
            return mu + eps * std
        else:
            # FIX 3: Added missing return statement
            return mu 
            
    def decode(self, z):
        x = self.from_bottleneck(z)                 # (B, 8192)
        x = x.view(-1, 32, 32, 8)                   # (B, C=32, H=32, W=8)
        x = self.decoder(x)                         # (B, 3, 512, 128)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)                         
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)                      
        return recon, mu, log_var # Good practice to return mu and log_var for the loss function!