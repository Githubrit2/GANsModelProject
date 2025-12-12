import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from models.dit_gan import DiTGenerator

def generate_images(model_path='generator_epoch_20.pth', num_images=16):
    """Generate fake images using trained DiT-GAN"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    generator = DiTGenerator(img_size=128, hidden_size=128, num_layers=3, latent_dim=256).to(device)
    generator.load_state_dict(torch.load(f'../outputs/{model_path}', map_location=device))
    generator.eval()
    
    # Generate images
    with torch.no_grad():
        noise = torch.randn(num_images, 256).to(device)
        fake_images = generator(noise)
        
        # Denormalize
        fake_images = (fake_images + 1) / 2
        
        # Plot results
        rows = int(np.sqrt(num_images))
        cols = (num_images + rows - 1) // rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
            
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < num_images:
                    img = fake_images[idx].cpu().permute(1, 2, 0).numpy()
                    axes[i][j].imshow(np.clip(img, 0, 1))
                axes[i][j].axis('off')
        
        plt.tight_layout()
        plt.savefig('../outputs/final_generated_images.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Generated {num_images} images saved as 'final_generated_images.png'")

if __name__ == "__main__":
    generate_images()