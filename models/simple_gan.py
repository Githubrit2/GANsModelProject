import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_size=128):
        super().__init__()
        
        # Correct interpretation of error:
        # [256, 128, 4, 4] means: out_channels=256, in_channels=128
        # [128, 64, 4, 4] means: out_channels=128, in_channels=64  
        # [64, 32, 4, 4] means: out_channels=64, in_channels=32
        # [32, 3, 4, 4] means: out_channels=32, in_channels=3
        
        self.model = nn.Sequential(
            # Linear: 100 -> 16384
            nn.Linear(latent_dim, 16384),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),  # 256*8*8 = 16384
            
            # ConvTranspose: 256 -> 128 channels (swapped to match checkpoint)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),  # BatchNorm matches output channels
            nn.ReLU(True),
            
            # ConvTranspose: 128 -> 64 channels  
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # ConvTranspose: 64 -> 32 channels
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # ConvTranspose: 32 -> 3 channels (final RGB)
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

class SimpleDiscriminator(nn.Module):
    def __init__(self, img_size=128):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 8, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)