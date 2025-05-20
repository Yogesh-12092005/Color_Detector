from flask import Flask, request, jsonify, render_template_string, send_from_directory
import cv2
import pandas as pd
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load extended color dataset
color_data = pd.read_csv('colors.csv')

# Improved HTML with basic UX additions
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Color Detection Tool</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

  body {
    font-family: 'Poppins', sans-serif;
    margin: 2rem;
    background: linear-gradient(to right, #dbeafe, #f0f9ff); /* Soft gradient */
    text-align: center;
    color: #222;
  }

  h1 {
    color: #1e3a8a;
    font-weight: 700;
    margin-bottom: 2rem;
  }

  input[type=file], button {
    font-size: 1rem;
    margin: 0.5rem;
    padding: 0.6rem 1rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    background-color: #3b82f6;
    color: white;
    transition: background-color 0.3s ease;
  }

  button:hover, input[type=file]:hover {
    background-color: #2563eb;
  }

  #canvas {
    margin-top: 1rem;
    border: 2px solid #ccc;
    max-width: 100%;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
  }

  #colorInfo {
    margin-top: 1.5rem;
    padding: 1rem;
    display: none;
    font-size: 1.2rem;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  }

  #colorDisplay {
    display: inline-block;
    width: 120px;
    height: 120px;
    margin-top: 0.5rem;
    border: 2px solid #555;
    background: #fff;
    border-radius: 12px;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.1);
  }

  #loading {
    display: none;
    font-style: italic;
    margin-top: 1rem;
    color: #444;
  }
</style>

</head>
<body>
  <h1>Color Detection Tool</h1>
  <input type="file" id="imageUpload" accept="image/*" />
  <br />
  <canvas id="canvas"></canvas>
  <br />
  <button id="detectBtn" disabled>Detect Color</button>
  <div id="loading">Detecting color...</div>
  <div id="colorInfo" style="display:none;">
    <strong>Color Name:</strong> <span id="colorName"></span><br />
    <strong>RGB:</strong> <span id="rgbValue"></span><br />
    <strong>Pixel Coordinates:</strong> <span id="pixelCoords"></span><br />
    <div id="colorDisplay"></div>
  </div>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const imageUpload = document.getElementById('imageUpload');
const detectBtn = document.getElementById('detectBtn');
const colorInfo = document.getElementById('colorInfo');
const colorNameSpan = document.getElementById('colorName');
const rgbValueSpan = document.getElementById('rgbValue');
const colorDisplay = document.getElementById('colorDisplay');
const pixelCoords = document.getElementById('pixelCoords');
const loading = document.getElementById('loading');

let img = new Image();
let clickX = null;
let clickY = null;
let uploadedImageUrl = null;

imageUpload.addEventListener('change', function(event) {
  const file = event.target.files[0];
  if(!file) return;
  const formData = new FormData();
  formData.append('image', file);

  fetch('/upload', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
      uploadedImageUrl = data.url + '?t=' + new Date().getTime(); // prevent caching
      img.src = uploadedImageUrl;
      img.onload = function() {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        colorInfo.style.display = 'none';
        detectBtn.disabled = true;
        clickX = null;
        clickY = null;
      }
    })
    .catch(err => alert('Upload failed'));
});

canvas.addEventListener('click', function(event) {
  if(img.width === 0 || img.height === 0) return;
  const rect = canvas.getBoundingClientRect();
  clickX = Math.floor(event.clientX - rect.left);
  clickY = Math.floor(event.clientY - rect.top);
  detectBtn.disabled = false;
  // Visual cue for selected point
  ctx.drawImage(img, 0, 0); // reset
  ctx.beginPath();
  ctx.arc(clickX, clickY, 5, 0, 2 * Math.PI);
  ctx.strokeStyle = 'red';
  ctx.lineWidth = 2;
  ctx.stroke();
});

detectBtn.addEventListener('click', function() {
  if(clickX === null || clickY === null) {
    alert('Please click on the image to select a point for color detection.');
    return;
  }
  loading.style.display = 'block';
  fetch('/detect_color', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: clickX, y: clickY})
  })
  .then(res => res.json())
  .then(data => {
    loading.style.display = 'none';
    if(data.error) {
      alert('Error: ' + data.error);
      return;
    }
    colorNameSpan.textContent = data.color_name;
    rgbValueSpan.textContent = '(' + data.r + ', ' + data.g + ', ' + data.b + ')';
    pixelCoords.textContent = '(' + clickX + ', ' + clickY + ')';
    colorDisplay.style.backgroundColor = 'rgb(' + data.r + ',' + data.g + ',' + data.b + ')';
    colorInfo.style.display = 'block';
  })
  .catch(() => {
    loading.style.display = 'none';
    alert('Failed to detect color');
  });
});
</script>
</body>
</html>
"""

def find_closest_color(pixel_rgb):
    rgb = np.array(pixel_rgb)
    dataset = color_data[['r', 'g', 'b']].to_numpy()
    distances = np.linalg.norm(dataset - rgb, axis=1)
    min_idx = distances.argmin()
    closest = color_data.iloc[min_idx]
    return {
        'color_name': closest['color_name'],
        'r': int(closest['r']),
        'g': int(closest['g']),
        'b': int(closest['b'])
    }

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return jsonify({'message': 'Image uploaded', 'url': '/' + filepath.replace('\\', '/')})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect_color', methods=['POST'])
def detect_color():
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    if x is None or y is None:
        return jsonify({'error': 'Coordinates not provided'}), 400
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not uploaded_files:
        return jsonify({'error': 'No image uploaded'}), 400
    latest_image = max(
        [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in uploaded_files],
        key=os.path.getctime
    )
    image = cv2.imread(latest_image)
    if image is None:
        return jsonify({'error': 'Failed to load image'}), 500
    height, width, _ = image.shape
    if y < 0 or y >= height or x < 0 or x >= width:
        return jsonify({'error': 'Coordinates out of image bounds'}), 400
    b, g, r = image[y, x]
    result = find_closest_color((r, g, b))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
