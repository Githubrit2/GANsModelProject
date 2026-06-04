# Web Application Setup

## HTML/CSS/JS Frontend with Flask Backend

### Installation

1. Install Flask:
```bash
pip install flask
```

### Running the Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

### Features
- Modern HTML/CSS/JS interface
- Real-time face generation
- Download generated images
- Responsive design
- No Streamlit dependency

### Project Structure
```
├── app.py              # Flask backend
├── templates/
│   └── index.html      # HTML frontend
├── static/
│   ├── style.css       # Styling
│   └── script.js       # JavaScript logic
└── web_app.py          # Old Streamlit app (deprecated)
```

### API Endpoint
- **POST /generate** - Generate face from text input
  - Request: `{"text": "your description"}`
  - Response: `{"image": "base64_data", "seed": 123456, "text": "your description"}`
