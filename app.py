import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("YOLO Nesne Tespiti Uygulaması")

# Kullanıcının model seçimi
model_option = st.selectbox(
    "Bir YOLO modeli seçin:",
    ["YOLOv9 Small (yolov9s.pt)", "YOLOv10 Small (yolov10s.pt)", "YOLO11 Small (yolo11s.pt)"]
)

# Seçilen modele göre yükleme
@st.cache_resource  
def load_model(model_name):
    return YOLO(model_name)

model_dict = {
    "YOLOv8 Small (yolov8s.pt)": "yolov9s.pt",
    "YOLOv8 Medium (yolov8m.pt)": "yolov10s.pt",
    "YOLOv8 Large (yolov8l.pt)": "yolo11s.pt"
}

model = load_model(model_dict[model_option])

uploaded_file = st.file_uploader("Bir PNG dosyası yükleyin", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)
    
    # Nesne tespiti yap
    st.write("### Nesne Tespiti Yapılıyor...")
    results = model.predict(image)
    
    # Tespit edilen nesneleri göster
    result_image = results[0].plot()
    st.image(result_image, caption='Tespit Sonucu', use_column_width=True)
