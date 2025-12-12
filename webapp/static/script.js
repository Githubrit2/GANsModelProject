let currentImageData = null;
let currentSeed = null;

async function generateFace() {
    const textInput = document.getElementById('textInput').value.trim();
    
    if (!textInput) {
        alert('⚠️ Please enter some text first!');
        return;
    }
    
    // Show loader
    document.getElementById('loader').style.display = 'flex';
    document.getElementById('generateBtn').disabled = true;
    
    try {
        console.log('Sending request with text:', textInput);
        
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: textInput })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display result
        currentImageData = data.image;
        currentSeed = data.seed;
        
        document.getElementById('generatedImage').src = data.image;
        document.getElementById('inputText').textContent = data.text;
        document.getElementById('seedValue').textContent = data.seed;
        document.getElementById('resultSection').style.display = 'block';
        
        console.log('Image displayed successfully');
        
    } catch (error) {
        console.error('Error:', error);
        alert('❌ Error generating image: ' + error.message);
    } finally {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('generateBtn').disabled = false;
    }
}

function downloadImage() {
    if (!currentImageData) return;
    
    const link = document.createElement('a');
    link.href = currentImageData;
    link.download = `generated_face_${currentSeed}.png`;
    link.click();
}

// Allow Enter key to generate
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('textInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            generateFace();
        }
    });
});
