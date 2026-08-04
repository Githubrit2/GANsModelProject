# Fake Image Generation using GANs (2025)

A modern implementation of GANs for face generation with clean organized codebase.

## 📁 Project Structure

```
├── models/                  # Model architectures
│   ├── simple_gan.py       # Simple CNN-based GAN
│   └── dit_gan.py          # Diffusion Transformer GAN
├── scripts/                 # Training & generation scripts
│   ├── train_simple_gan.py # Train Simple GAN
│   ├── train_dit_gan.py    # Train DiT-GAN
│   └── generate_images.py  # Generate images
├── webapp/                  # Web application
│   ├── flask_app.py        # Flask backend
│   ├── templates/          # HTML files
│   └── static/             # CSS & JS
├── docs/                    # Documentation
├── outputs/                 # Generated images & models
├── ffhq_small/             # Dataset (3K images)
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
cd scripts
python train_simple_gan.py
```

### 3. Run Web App
```bash
cd webapp
python flask_app.py
```


Open: http://localhost:5000

## 🎯 Features
- **Two GAN Architectures**: Simple CNN & DiT-based
- **Web Interface**: Modern HTML/CSS/JS frontend
- **FFHQ Dataset**: 3,000 high-quality faces
- **CPU Optimized**: Efficient training

## 📊 Training Parameters
- Epochs: 100
- Batch Size: 4 (CPU) / 16 (GPU)
- Image Size: 128x128
- Latent Dim: 100

## 🔧 Usage

### Training
```bash
cd scripts
python train_simple_gan.py  # Simple GAN
python train_dit_gan.py     # DiT-GAN
```

### Generation
```bash
cd scripts
python generate_images.py
```

### Web App
```bash
python webapp\flask_app.py
```

============