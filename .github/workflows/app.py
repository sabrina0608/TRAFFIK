import streamlit as st
import pickle
import numpy as np
import os
from pathlib import Path

# 1. Sahifa sozlamalari va Dizayn
st.set_page_config(
    page_title="Trafik Bashorat Tizimi",
    page_icon="🚦",
    layout="wide"
)

# Estetik ko'rinish uchun CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #2e7d32; color: white; height: 3em; font-weight: bold; }
    .footer { position: fixed; bottom: 0; width: 100%; text-align: center; color: grey; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# 2. Modelni yuklash funksiyasi
@st.cache_resource
def load_traffic_model():
    # Fayl nomi siz aytgandek: "model (2).pkl"
    file_name = "model (2).pkl"
    base_path = Path(__file__).parent
    file_path = base_path / file_name
    
    if file_path.exists():
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Modelni o'qishda xatolik: {e}")
            return None
    else:
        st.error(f"❌ '{file_name}' fayli topilmadi!")
        st.info(f"Hozirgi papka: {os.listdir(base_path)}") # Diagnostika uchun
        return None

# Modelni chaqiramiz
model = load_traffic_model()

# 3. Asosiy Interfeys
st.title("🚦 Aqlli Trafik Bashorat Tizimi")
st.subheader("Trafik holatini aniqlash uchun ko'rsatkichlarni kiriting")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Transport Ma'lumotlari")
        cars = st.number_input("Avtomobillar soni", min_value=0, step=1, help="Yo'ldagi jami mashinalar soni")
        buses = st.number_input("Avtobuslar soni", min_value=0, step=1)
        bikes = st.number_input("Mototsikl/Velosipedlar soni", min_value=0, step=1)
        
    with col2:
        st.markdown("### 📅 Vaqt va Sharoit")
        hour = st.slider("Kun vaqti (Soat)", 0, 23, 12)
        day_of_week = st.selectbox("Hafta kuni", 
                                  ["Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba", "Yakshanba"])
        weather = st.selectbox("Ob-havo sharoiti", ["Ochiq", "Yomg'ir", "Qor", "Tuman"])

# 4. Bashorat qilish qismi
st.divider()

if st.button("Trafik holatini hisoblash"):
    if model is not None:
        try:
            # Modelga kiruvchi ma'lumotlarni tayyorlash
            # DIQQAT: Model qanday tartibda o'qitilgan bo'lsa, xususiyatlarni shunday uzatish kerak
            # Quyida misol tariqasida 4 ta xususiyat berilgan:
            input_data = np.array([[cars, buses, bikes, hour]]) 
            
            prediction = model.predict(input_data)
            
            # Natijani ko'rsatish
            st.success("### Bashorat natijasi:")
            if prediction[0] == 1: # Agar model klassifikatsiya bo'lsa
                st.warning("⚠️ Yo'lda tirbandlik ehtimoli yuqori!")
            else:
                st.info("✅ Yo'l ochiq, harakatlanish barqaror.")
                
            st.metric(label="Trafik indeksi", value=f"{prediction[0]}")
            st.balloons()
            
        except Exception as e:
            st.error(f"Bashorat qilishda xatolik: {e}. Model kutayotgan ustunlar soni mos kelmasligi mumkin.")
    else:
        st.error("Model yuklanmaganligi sababli hisoblab bo'lmaydi.")

# 5. Mualliflik huquqi va Footer
st.markdown(f"""
    <div class="footer">
        <hr>
        <p><b>Dastur egasi:</b> Sayfiddinova Sabrina</p>
        <p>Samarqand davlat universiteti</p>
    </div>
""", unsafe_allow_html=True)
