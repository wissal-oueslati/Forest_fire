import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from ultralytics import YOLO
# Load your trained model
from tensorflow.keras.models import load_model
import os
from PIL import Image
import uuid

# Définissez le chemin de votre image de fond
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://wallpaperaccess.com/full/3964713.jpg");
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

import requests
headers = {"Authorization": "Bearer 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"}
API_URL1 = "https://api-inference.huggingface.co/models/dima806/car_brand_image_detection"
API_URL2 = "https://api-inference.huggingface.co/models/beingamit99/car_damage_detection"


def query(filename, API_URL, timeout=30):
    with open(filename, "rb") as f:
        data = f.read()
    try:
        response = requests.post(API_URL, headers=headers, data=data, timeout=timeout)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None  # You can modify this part based on your error handling logic

car_types = {
    'Aston Martin': 'Luxury',
    'Mercedes-Benz': 'Luxury',
    'Mini': 'Standard',
    'Tesla': 'Electric',
    'GMC': 'SUV',
    'Alfa Romeo': 'Sport',
    'Studebaker': 'Classic',
    'Suzuki': 'Standard',
    'Peugeot': 'Standard',
    'Genesis': 'Luxury',
    'BMW': 'Luxury',
    'Honda': 'Standard',
    'Chrysler': 'Standard',
    'Mazda': 'Standard',
    'Infiniti': 'Luxury',
    'Land Rover': 'SUV',
    'Dodge': 'Standard',
    'Fiat': 'Standard',
    'Maserati': 'Luxury',
    'Saab': 'Standard',
    'Nissan': 'Standard',
    'Hudson': 'Classic',
    'Lincoln': 'Luxury',
    'Volvo': 'Luxury',
    'Mitsubishi': 'Standard',
    'Oldsmobile': 'Classic',
    'Lexus': 'Luxury',
    'Buick': 'Luxury',
    'Jaguar': 'Luxury',
    'Toyota': 'Standard',
    'Volkswagen': 'Standard',
    'Renault': 'Standard',
    'Citroen': 'Standard',
    'Audi': 'Luxury',
    'Subaru': 'Standard',
    'Cadillac': 'Luxury',
    'Pontiac': 'Standard',
    'Porsche': 'Sport',
    'Daewoo': 'Standard',
    'Bugatti': 'Exotic',
    'Jeep': 'SUV',
    'Ram Trucks': 'Truck',
    'Chevrolet': 'Standard',
    'MG': 'Sport',
    'Hyundai': 'Standard',
    'Ferrari': 'Exotic',
    'Acura': 'Luxury',
    'Kia': 'Standard',
    'Bentley': 'Luxury',
    'Ford': 'Standard',
}
repair_cost_by_type = {
    'Luxury': {
        'dent': {'01-minor': 500, '02-moderate': 1000, '03-severe': 2000},
        'scratch': {'01-minor': 300, '02-moderate': 700, '03-severe': 1500},
        'crack': {'01-minor': 800, '02-moderate': 1200, '03-severe': 2500},
        'glass Shatter': {'01-minor': 1000, '02-moderate': 1800, '03-severe': 3000},
        'lamp broken': {'01-minor': 600, '02-moderate': 1000, '03-severe': 2000},
        'tire flat' :{'01-minor': 200, '02-moderate': 400, '03-severe': 800},
    },
    'Standard': {
        'dent': {'01-minor': 400, '02-moderate': 850, '03-severe': 1600},
        'scratch': {'01-minor': 200, '02-moderate': 550, '03-severe': 1200},
        'crack': {'01-minor': 700, '02-moderate': 1050, '03-severe': 2200},
        'glass Shatter': {'01-minor': 800, '02-moderate': 1500, '03-severe': 2700},
        'lamp broken': {'01-minor': 500, '02-moderate': 900, '03-severe': 1800},
        'tire flat' :{'01-minor': 150, '02-moderate': 300, '03-severe': 600},
    },
    'Sport': {
        'dent': {'01-minor': 600, '02-moderate': 1100, '03-severe': 2100},
        'scratch': {'01-minor': 350, '02-moderate': 800, '03-severe': 1600},
        'crack': {'01-minor': 900, '02-moderate': 1300, '03-severe': 2600},
        'glass Shatter': {'01-minor': 1100, '02-moderate': 2000, '03-severe': 3200},
        'lamp broken': {'01-minor': 700, '02-moderate': 1200, '03-severe': 2300},
        'tire flat' :{'01-minor': 500, '02-moderate': 700, '03-severe': 1000},
    },
    'Electric': {
        'dent': {'01-minor': 700, '02-moderate': 1200, '03-severe': 2300},
        'scratch': {'01-minor': 400, '02-moderate': 900, '03-severe': 1800},
        'crack': {'01-minor': 1000, '02-moderate': 1400, '03-severe': 2700},
        'glass Shatter': {'01-minor': 1200, '02-moderate': 2200, '03-severe': 3400},
        'lamp broken': {'01-minor': 800, '02-moderate': 1300, '03-severe': 2400},
        'tire flat' :{'01-minor': 300, '02-moderate': 400, '03-severe': 600},
    },
    'SUV': {
        'dent': {'01-minor': 500, '02-moderate': 950, '03-severe': 1800},
        'scratch': {'01-minor': 250, '02-moderate': 600, '03-severe': 1300},
        'crack': {'01-minor': 800, '02-moderate': 1100, '03-severe': 2200},
        'glass Shatter': {'01-minor': 900, '02-moderate': 1600, '03-severe': 2800},
        'lamp broken': {'01-minor': 550, '02-moderate': 1000, '03-severe': 2000},
        'tire flat' :{'01-minor': 200, '02-moderate': 600, '03-severe': 800},
    },
    'Classic': {
        'dent': {'01-minor': 300, '02-moderate': 700, '03-severe': 1500},
        'scratch': {'01-minor': 150, '02-moderate': 500, '03-severe': 1100},
        'crack': {'01-minor': 600, '02-moderate': 1000, '03-severe': 2000},
        'glass Shatter': {'01-minor': 700, '02-moderate': 1300, '03-severe': 2500},
        'lamp broken': {'01-minor': 400, '02-moderate': 800, '03-severe': 1700},
        'tire flat' :{'01-minor': 150, '02-moderate': 300, '03-severe': 500},
    },
    # Add more types and their repair cost details...
}


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
    class_names=[]
    # Extract boxes and plot the results
    boxes = results[0].boxes
    res_plotted = results[0].plot()[:, :, ::-1]
    for box in results[0].boxes:
    #extract the label name
        label=model3.names.get(box.cls.item())
        class_names.append(label)

    return boxes, res_plotted, class_names

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
                    st.write(
                        f"""
                        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px;">
                            <p style="color: black;">Severity class: {pred_label}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if st.button("Detect Damage"):
                    _, damage_result, class_names = detect_damage(uploaded_image)
                    damage_result_placeholder.image(damage_result, caption='Detected Image')
                    uploaded_image_placeholder.empty()  # Clear the uploaded image placeholder
                    print(class_names)
                

                severity_result = predict_severity(uploaded_image)
                pred_label = class_names[np.argmax(severity_result)]

                st.write("Choose a car type:")

                # Create a dropdown list of car types
                option = st.selectbox("", 
                                    list(car_types.keys()),
                                    index=None,
                                    )
                st.markdown(f"""
                            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px;">
                                <p style="color: black;">You selected: {option} - Type: {car_types[option]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True)
                car_type=car_types[option]
                print (car_type)
                
                if st.button("cost estimation"):   
                    damage = "damages are: "
                    total_cost=0
                    _, damage_result, class_names = detect_damage(uploaded_image)
                    for value in class_names:
                        cost = repair_cost_by_type[car_type][value][pred_label]
                        print(cost)
                        damage += str(value)+" : " +str(repair_cost_by_type[car_type][value][pred_label])+", " 
                        print (damage)
                        total_cost += repair_cost_by_type[car_type][value][pred_label]
                        print(total_cost)
                

                    car_info_html = (
                        f"""
                            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px;">
                                <p style="color: black;">{damage}</p>
                                <p style="color: black;">The total cost of repair is : {total_cost}</p>
                                
                            </div>
                        """)

                    st.markdown(car_info_html, unsafe_allow_html=True)


