import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("YOLO Nesne Tespiti Uygulaması")

@st.cache_resource  
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

uploaded_file = st.file_uploader("Bir PNG dosyası yükleyin", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)
    
    # Nesne tespiti yap
    st.write("### Nesne Tespiti Yapılıyor...")
    results = model.predict(image)
    boxes = results.boxes
    labels = results.names
    # labels, her nesne için bir sayıyı verir. Örneğin, [0, 1, 2] gibi.

    # Tanınan nesneleri ve sayısını saymak için:
    object_counts = {}

    for label in boxes.cls:
        # labels'dan nesne etiketine dönüştür
        label_name = labels[int(label)]
        if label_name not in object_counts:
            object_counts[label_name] = 1
        else:
            object_counts[label_name] += 1
    
    # Tespit edilen nesneleri göster
    result_image = results[0].plot()
    st.image(result_image, caption='Tespit Sonucu', use_column_width=True)

    st.write("Detected objects and their counts:")
    for obj, count in object_counts.items():
        st.write(f"{obj}: {count}")
