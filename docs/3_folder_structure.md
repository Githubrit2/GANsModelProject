# Project Structure Guide

## Directory Organization

### `/models`
Contains all GAN model architectures:
- `simple_gan.py` - CNN-based Generator & Discriminator
- `dit_gan.py` - Transformer-based DiT Generator
- `__init__.py` - Package initialization

### `/scripts`
Training and utility scripts:
- `train_simple.py` - Train Simple GAN
- `train.py` - Train DiT-GAN
- `generate.py` - Generate images from trained models
- `reduce_dataset.py` - Dataset preprocessing
- `presentation_demo.py` - Demo script

### `/webapp`
Web application files:
- `app.py` - Flask backend server
- `web_app.py` - Old Streamlit app (deprecated)
- `/templates` - HTML files
- `/static` - CSS & JavaScript

### `/outputs`
Generated files (gitignored):
- `*.pth` - Model checkpoints
- `*.png` - Generated images
- `/gan_results` - Training results

### `/docs`
Documentation files:
- `PROJECT_DOCUMENTATION.md` - Full documentation
- `README_WEBAPP.md` - Web app guide
- `/UML Diagrams` - Architecture diagrams

### `/ffhq_small`
Dataset folder (3,000 FFHQ images)

## Import Paths

From scripts:
```python
import sys
sys.path.append('..')
from models.simple_gan import SimpleGenerator
```

From webapp:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.simple_gan import SimpleGenerator
```

## Running Components

**Training:**
```bash
cd scripts
python train_simple.py
```

**Web App:**
```bash
cd webapp
python app.py
```

Or use: `run_webapp.bat`
