# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import os
from keras.utils import load_img, img_to_array
from keras.models import load_model

from PIL import Image

filepath = 'E:\Downloads\Plant-Leaf-Disease-Prediction-main\leaf_disease_model.h5'
model = load_model(filepath)
print("Model Loaded Successfully")

def pred_tomato_disease(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))  # Load the image
    print("@@ Got Image for prediction")
  
    test_image = img_to_array(test_image) / 255.0  # Convert image to numpy array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions from 3D to 4D
  
    result = model.predict(test_image)  # Perform prediction
    print('@@ Raw result = ', result)
  
    pred = np.argmax(result, axis=1)[0]  # Get the predicted class index
    print(pred)
  
    class_labels = [
        "Tomato - Bacteria Spot",
        "Tomato - Early Blight",
        "Tomato - Late blight",
        "Tomato - Leaf Mold",
        "Tomato - Septoria leaf spot",
        "Tomato - Two spotted spider mite",
        "Tomato - Target Spot",
        "Tomato - Tomato Yellow Leaf Curl Virus",
        "Tomato - Tomato mosaic virus",
        "Tomato - Healthy"
    ]
  
    output_page = "Tomato_-_Healthy.html"  # Default output page is for healthy tomato
  
    if pred >= 0 and pred < len(class_labels):
        output_page = class_labels[pred].replace(" ", "_") + ".html"
    
    return class_labels[pred], output_page

# Create flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Get input image from client, predict class, and render respective .html page for the solution
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # Get the input file
        filename = file.filename        
        print("@@ Input posted =", filename)
        
        file_path = os.path.join('static/upload', filename)
        file.save(file_path)
        user_image1 = os.path.join('upload/', filename)

        print("@@ Predicting class...")
        pred, output_page = pred_tomato_disease(tomato_plant=file_path)
              
        return render_template(output_page, pred_output=pred, user_image=user_image1)

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=8080)

    
    
