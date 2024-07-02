import os
from flask import Flask, render_template, request
import joblib
from PIL import Image
import numpy as np
import tensorflow as tf


app = Flask(__name__)

# Load your trained ML model
print(os.getcwd())
model = tf.keras.models.load_model("C:\\Users\\Asus\\OneDrive\\Desktop\\nigga\\leaflens.github.io\model (1).h5")
# model.summary()
classes = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']
@app.route('/')
def index():
    return render_template('index.html', result='')

@app.route('/your_ml_model_endpoint', methods=['POST'])
def predict():
    # Get the uploaded file
    try:
        uploaded_file = request.files['file']
    except KeyError:
        return render_template('index.html', result='No file uploaded.')


    if uploaded_file.filename != '':
        
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist

        image_path = os.path.join(upload_dir, uploaded_file.filename)
        
        print(image_path)
        uploaded_file.save(image_path)
        # Preprocess the image (resize, convert to numpy array, etc.)
        img = Image.open(image_path)
        img = img.resize((128, 128))  # Adjust to your model's input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add the batch dimension
        img_array = img_array / 255.0  # Normalize if needed

        # Make a prediction using your model
        prediction_values = model.predict([img_array])

        predicted_class = classes[np.argmax(prediction_values)]

        return render_template('index.html', result=f'Predicted Plant: {predicted_class}')

    return render_template('index.html', result='No file uploaded.')

if __name__ == '__main__':
    app.run(debug=True)
