import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# YOLO modelini yükle
model = YOLO("yolo11s.pt")  # Model dosyasının doğru yolunu belirtin

st.title("YOLO Nesne Tespiti")

# Görüntü yükleme alanı
uploaded_file = st.file_uploader("Bir PNG dosyası yükleyin", type=["png"])

if uploaded_file is not None:
    # Yüklenen dosyayı PIL ile aç
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

    # Model ile nesne tespiti yap
    results = model.predict(np.array(image))

    # Sonuçları göster
    st.image(results[0].plot(), caption='Nesne Tespiti Sonucu', use_column_width=True)

# Kameradan canlı görüntü alma
st.subheader("Kameradan Nesne Tespiti")

# Kamera ile görüntü alma
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı ifade eder

stframe = st.empty()  # Streamlit'te görüntüyü göstereceğimiz alan

if st.button("Kamerayı Başlat"):
    while True:
        ret, frame = cap.read()

        if not ret:
            st.write("Kamera hatası!")
            break
        
        # Görüntüyü BGR'den RGB'ye dönüştür
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Model ile nesne tespiti yap
        results = model.predict(img_rgb)

        # Sonuçları çiz
        img_result = results[0].plot()  # YOLO sonucunu çiz

        # Görüntüyü Streamlit'te göster
        stframe.image(img_result, channels="RGB", use_column_width=True)

        # Eğer "Stop Video" butonuna basılırsa, video durdurulacak
        if st.button("Stop Video"):
            break

    # Kamerayı serbest bırak
    cap.release()
    cv2.destroyAllWindows()
