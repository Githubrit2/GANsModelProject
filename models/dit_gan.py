import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiTBlock(nn.Module):
    """Diffusion Transformer Block"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class DiTGenerator(nn.Module):
    """DiT-based Generator"""
    def __init__(self, img_size=256, patch_size=16, hidden_size=512, num_layers=12, num_heads=8, latent_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, self.num_patches * hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, patch_size * patch_size * 3)
        
    def forward(self, noise):
        B = noise.shape[0]
        
        # Project noise to patch embeddings
        x = self.input_proj(noise)  # B, num_patches * hidden_size
        x = x.view(B, self.num_patches, self.hidden_size)  # B, num_patches, hidden_size
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Decode to image
        x = self.norm(x)
        x = self.decoder(x)
        
        # Reshape to image
        x = x.view(B, self.num_patches, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, num_patches, patch_size, patch_size
        
        # Reconstruct image from patches
        patches_per_row = self.img_size // self.patch_size
        x = x.view(B, 3, patches_per_row, patches_per_row, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, 3, self.img_size, self.img_size)
        
        return torch.tanh(x)

class Discriminator(nn.Module):
    """Simple CNN Discriminator"""
    def __init__(self, img_size=256):
        super().__init__()
        if img_size == 128:
            self.model = nn.Sequential(
                # 128x128 -> 64x64
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                
                # 64x64 -> 32x32
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                # 32x32 -> 16x16
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                
                # 16x16 -> 8x8
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                
                # 8x8 -> 1x1
                nn.Conv2d(512, 1, 8, 1, 0),
                nn.Sigmoid()
            )
        else:  # 256x256
            self.model = nn.Sequential(
                # 256x256 -> 128x128
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                
                # 128x128 -> 64x64
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                # 64x64 -> 32x32
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                
                # 32x32 -> 16x16
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                
                # 16x16 -> 8x8
                nn.Conv2d(512, 1024, 4, 2, 1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2),
                
                # 8x8 -> 1x1
                nn.Conv2d(1024, 1, 8, 1, 0),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        return self.model(x).view(-1, 1)