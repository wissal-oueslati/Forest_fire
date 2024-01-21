import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from ultralytics import YOLO
# Load your trained model
from tensorflow.keras.models import load_model

# Définissez le chemin de votre image de fond
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://c4.wallpaperflare.com/wallpaper/432/546/410/clouds-field-old-car-vintage-wallpaper-preview.jpg");
background-size: cover;

}
[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}

</style>

"""
st.markdown(page_bg_img, unsafe_allow_html=True) 

nav_selection = st.sidebar.radio("Navigation", ["Overview", "AI Check", "Car Diagnostic"])
# Overview section
if nav_selection == "Overview":
    st.markdown(
        """
        <div style='text-align: center; background-color: white; color: black; border-radius: 10px; padding: 20px;'>
            <h2 style='color: black;'>Welcome to the Car Damage Detection App!</h2>
            <p>This application employs cutting-edge AI models to analyze uploaded car images and provide insightful information regarding potential damages.</p>
            <h3 style='color: black;'>Functionalities:</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li>AI Check: Determine the authenticity of the uploaded car image.</li>
                <li>Severity Classification: Classify the severity of damages in the image.</li>
                <li>Damage Detection: Detect and highlight damages within the uploaded image.</li>
                <li>Damage Cost Prediction: Estimate the repair cost based on identified damages.</li>
            </ul>
            <h3 style='color: black;'>How to Use:</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li>Upload an image of a damaged car in JPG, PNG, or JPEG format.</li>
                <li>Choose one of the available options from the sidebar to obtain specific insights about the uploaded image.</li>
            </ul>
            <p style='font-style: italic;'>This application uses advanced machine learning models to deliver accurate and rapid results. The provided information aims to assist in assessing car damages but should be used as a reference only. We're constantly improving the app, and the "Damage Cost Prediction" feature will be available soon.</p>
            <p>Thank you for using the Car Damage Detection App!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    
# load models
model1 = load_model('d:/bureau/damaged_car/models/ai_or_not/model2.h5')  # Replace with the actual path
model2 = load_model('d:/bureau/damaged_car/models/severity_class/resnet50.h5') 
model3 = YOLO('d:/bureau/damaged_car/models/detection/yolo/working2/weights/best.pt')

# Function to make predictions and get the predicted label
def predict_image(image_path, threshold=0.5):
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.0  # Normalize the pixel values

    predictions = model1.predict(test_image)
    
    if predictions >= threshold:
        return "real"
    else:
        return "fake"

    
def predict_severity(image_path):
    test_image = image.load_img(image_path, target_size=(150,150))  # Redimensionner l'image à la taille attendue
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.0  # Normalisation des valeurs de pixels

    predictions = model2.predict(test_image)
    return predictions

def detect_damage(image_path):
    test_image = image.load_img(image_path, target_size=(640, 640))
    results = model3.predict(test_image, conf=0.4)
    # Extract boxes and plot the results
    boxes = results[0].boxes
    res_plotted = results[0].plot()[:, :, ::-1]

    return boxes, res_plotted

# Variables pour stocker les résultats
result = {
    "AI": None,
    "Severity": None,
    "Damage": None
}
# ...

if nav_selection == "AI Check" or nav_selection == "Car Diagnostic":
    st.title("Car Damage Detection App")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Placeholder for displaying images and results
    col1, col2 = st.columns([1, 1])

    if nav_selection == "AI Check":
        with col1:
            uploaded_image_placeholder = st.empty()
            damage_result_placeholder = st.empty()
            if uploaded_image is not None:
                uploaded_image_placeholder.image(uploaded_image, caption="Uploaded Image")

        with col2:
            if uploaded_image is not None:
                if st.button("Test AI"):
                    ai_result = predict_image(uploaded_image)
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px;">
                            <p style="color: black;">This image is {ai_result}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    elif nav_selection == "Car Diagnostic":
        with col1:
            uploaded_image_placeholder = st.empty()
            damage_result_placeholder = st.empty()
            if uploaded_image is not None:
                uploaded_image_placeholder.image(uploaded_image, caption="Uploaded Image")

        with col2:
            if uploaded_image is not None:
                class_names = ["01-minor", "02-moderate", "03-severe"]

                if st.button("Severity Class"):
                    severity_result = predict_severity(uploaded_image)
                    pred_label = class_names[np.argmax(severity_result)]
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px;">
                            <p style="color: black;">Severity class: {pred_label}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if st.button("Detect Damage"):
                    _, damage_result = detect_damage(uploaded_image)
                    damage_result_placeholder.image(damage_result, caption='Detected Image')
                    uploaded_image_placeholder.empty()  # Clear the uploaded image placeholder
