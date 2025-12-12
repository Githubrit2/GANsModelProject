import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('..')
from models.simple_gan import SimpleGenerator, SimpleDiscriminator

class FFHQDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def train_simple_gan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 16
    lr = 0.0002
    num_epochs = 50
    latent_dim = 100
    img_size = 128
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = FFHQDataset('../ffhq_small', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Models
    generator = SimpleGenerator(latent_dim=latent_dim, img_size=img_size).to(device)
    discriminator = SimpleDiscriminator(img_size=img_size).to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        for i, real_images in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # Print progress
            if i % 20 == 0:
                print(f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        # Save sample images
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                noise = torch.randn(16, latent_dim).to(device)
                fake_images = generator(noise)
                
                # Denormalize
                fake_images = (fake_images + 1) / 2
                
                # Create grid
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx, ax in enumerate(axes.flat):
                    if idx < fake_images.size(0):
                        img = fake_images[idx].cpu().permute(1, 2, 0).numpy()
                        ax.imshow(np.clip(img, 0, 1))
                    ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'../outputs/simple_generated_epoch_{epoch+1}.png')
                plt.close()
        
        # Save models
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'../outputs/simple_generator_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train_simple_gan()