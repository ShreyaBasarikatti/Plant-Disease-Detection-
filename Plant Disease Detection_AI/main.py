import streamlit as st
import os
import io

import json
from PIL import Image

import numpy as np
import tensorflow as tf
import google.generativeai as genai
from keras.models import load_model
from keras.preprocessing.image import img_to_array , load_img
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

GOOGLE_API_KEY = "YOUR API KEY"  
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="PROJECT NAME",
    page_icon="🌱",
    layout="wide")

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/best_model.h5"
model = tf.keras.models.load_model(model_path)

data = [
    "apple apple scab",
    "apple black rot",
    "apple cedar apple rust",
    "apple healthy",
    "blueberry healthy",
    "cherry including sour powdery mildew",
    "cherry including sour healthy",
    "corn maize cercospora leaf spot gray leaf spot",
    "corn maize common rust",
    "corn maize northern leaf blight",
    "corn maize healthy",
    "grape black rot",
    "grape esca black measles",
    "grape leaf blight isariopsis leaf spot",
    "grape healthy",
    "orange haunglongbing citrus greening",
    "peach bacterial spot",
    "peach healthy",
    "pepper bell bacterial spot",
    "pepper bell healthy",
    "potato early blight",
    "potato late blight",
    "potato healthy",
    "raspberry healthy",
    "soybean healthy",
    "squash powdery mildew",
    "strawberry leaf scorch",
    "strawberry healthy",
    "tomato bacterial spot",
    "tomato early blight",
    "tomato late blight",
    "tomato leaf mold",
    "tomato septoria leaf spot",
    "tomato spider mites two spotted spider mite",
    "tomato target spot",
    "tomato tomato yellow leaf curl virus",
    "tomato tomato mosaic virus",
    "tomato healthy"
]

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

ref = {}
for idx, name in enumerate(data):
    ref[idx] = name

def predict_image_class(model, image_path, data):
    img = load_img(image_path , target_size=(224,224))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im , axis=0)
    pred = np.argmax(model.predict(img))
    disease=ref[pred]
    model = genai.GenerativeModel('gemini-pro')
    how = model.generate_content("YOUR PROMPT")
    response = model.generate_content("YOUR PROMPT")
    summary=response.text
    return disease,summary,how.text
   

st.title("🌱PROJECT TITLE")
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_img = image.resize((350, 350))
    st.image(resized_img)

    if st.button('Classify'):
        prediction,c,d = predict_image_class(model, uploaded_image, data)
        st.success(f'Prediction: {str(prediction)}')
        st.success(f'{str(d)}')
        st.success(f'{str(c)}')


        

    


    
