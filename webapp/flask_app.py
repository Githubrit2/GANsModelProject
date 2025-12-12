from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image
import io
import base64
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.simple_gan import SimpleGenerator

app = Flask(__name__)

# Print startup info
print("="*50)
print("Starting Flask App...")
print(f"Working Directory: {os.getcwd()}")
print(f"Model Path: {os.path.abspath('../outputs/generator_epoch_100.pth')}")
print("="*50)

# Load model
device = torch.device('cpu')
generator = SimpleGenerator(latent_dim=100, img_size=128).to(device)
model_loaded = False

try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'generator_epoch_100.pth')
    print(f"Loading model from: {model_path}")
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    model_loaded = True
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()

def text_to_seed(text):
    return hash(text) % 1000000

def generate_face(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    with torch.no_grad():
        noise = torch.randn(1, 100).to(device)
        fake_image = generator(noise)
        fake_image = (fake_image + 1) / 2
        
        img_array = fake_image[0].cpu().permute(1, 2, 0).numpy()
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        print("\n" + "="*50)
        print("Received generation request")
        
        if not model_loaded:
            print("[ERROR] Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        print(f"Request data: {data}")
        
        text = data.get('text', '')
        print(f"Text input: {text}")
        
        if not text.strip():
            print("[ERROR] Empty text")
            return jsonify({'error': 'Please enter text'}), 400
        
        seed = text_to_seed(text)
        print(f"Generated seed: {seed}")
        
        print("Generating face...")
        img = generate_face(seed)
        print("Face generated successfully")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        print(f"Image encoded (size: {len(img_str)} chars)")
        
        response = {
            'image': f'data:image/png;base64,{img_str}',
            'seed': seed,
            'text': text
        }
        print("[OK] Sending response")
        print("="*50 + "\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Error in generate(): {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
