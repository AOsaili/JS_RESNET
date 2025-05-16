from flask import Flask, request, render_template, jsonify
import os
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ResNet model
def init_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model

# Initialize preprocessing
def init_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load ImageNet labels
def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    return response.json()

# Initialize components
model = init_model()
preprocess = init_preprocess()
labels = get_imagenet_labels()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the image
        image_path = os.path.join(UPLOAD_FOLDER, 'captured.jpg')
        file.save(image_path)
        
        # Process image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)
        
        # Run prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 3)
        
        results = []
        for i in range(top_prob.size(0)):
            results.append({
                'label': labels[top_catid[i].item()],
                'probability': f"{top_prob[i].item():.2%}"
            })
        
        return jsonify({
            'top_prediction': results[0],
            'all_predictions': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)