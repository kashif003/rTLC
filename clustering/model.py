from torch import nn
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