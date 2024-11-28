import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("YOLO Object Detection Application")

# Model selection by the user
model_option = st.selectbox(
    "Choose a YOLO model:",
    ["YOLO9s", "YOLO10s", "YOLO11s"]
)

# Load the selected model
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model_files = {
    "YOLO9s": "yolov9s.pt",
    "YOLO10s": "yolov10s.pt",
    "YOLO11s": "yolo11s.pt"
}

model = load_model(model_files[model_option])

# Image upload section
uploaded_file = st.file_uploader("Upload a PNG file", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform object detection
    st.write("### Performing Object Detection...")
    results = model.predict(image)
    
    # Display results
    result_image = results[0].plot()  # Annotated image
    st.image(result_image, caption='Detection Result', use_column_width=True)
