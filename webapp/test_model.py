import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing model loading...")
print(f"Current directory: {os.getcwd()}")

try:
    from models.simple_gan import SimpleGenerator
    print("[OK] Model import successful")
    
    import torch
    device = torch.device('cpu')
    generator = SimpleGenerator(latent_dim=100, img_size=128).to(device)
    print("[OK] Generator created")
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'generator_epoch_100.pth')
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("[OK] Model loaded successfully")
    
    # Test generation
    import numpy as np
    noise = torch.randn(1, 100).to(device)
    with torch.no_grad():
        fake_image = generator(noise)
    print(f"[OK] Generated image shape: {fake_image.shape}")
    print("\n[SUCCESS] Everything works! You can run the Flask app now.")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
