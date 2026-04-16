import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# 1. Sahifa sozlamalari
st.set_page_config(page_title="Trafik Bashorat Pro", layout="wide", page_icon="🚦")

# Maxsus dizayn (CSS)
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-title { color: #1e3a8a; text-align: center; font-weight: bold; border-bottom: 2px solid #1e3a8a; padding-bottom: 10px; }
    .author-info { text-align: center; color: #555; margin-bottom: 30px; font-style: italic; }
    </style>
""", unsafe_allow_html=True)

# 2. Modelni yuklash (Kesh bilan)
@st.cache_resource
def load_model():
    # Fayl nomi siz aytgandek: model (2).pkl
    file_name = "model (2).pkl"
    file_path = Path(__file__).parent / file_name
    
    if file_path.exists():
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Model yuklashda xato: {e}")
    return None

model = load_model()

# 3. Sarlavha va Muallif
st.markdown("<h1 class='main-title'>🚦 Professional AI Bashorat Tizimi</h1>", unsafe_allow_html=True)
st.markdown("<p class='author-info'>Dastur egasi: Sayfiddinova Sabrina — Samarqand davlat universiteti</p>", unsafe_allow_html=True)

# 4. Sidebar: Ma'lumotlarni kiritish (7 ta xususiyat)
st.sidebar.header("📊 Kiruvchi parametrlar")
with st.sidebar:
    cars = st.number_input("Avtomobillar", 0, 2000, 100)
    buses = st.number_input("Avtobuslar", 0, 500, 20)
    bikes = st.number_input("Mototsikllar", 0, 1000, 50)
    hour = st.slider("Vaqt (0-23)", 0, 23, 14)
    day_of_week = st.selectbox("Hafta kuni", options=range(7), format_func=lambda x: ["Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba", "Yakshanba"][x])
    is_holiday = st.radio("Bayram kuni?", [0, 1], format_func=lambda x: "Ha" if x == 1 else "Yo'q")
    weather_index = st.slider("Ob-havo holati (1-Yaxshi, 4-Yomon)", 1, 4, 1)

# 5. Asosiy Interfeys (Tablar)
tab1, tab2, tab3 = st.tabs(["🎯 Bashorat", "📈 Analitika", "🔬 Regressiya Tahlili"])

with tab1:
    st.subheader("Model Bashorati")
    if st.button("Hisoblashni boshlash"):
        if model:
            try:
                # Model kutyotgan 7 ta xususiyatni uzatish
                input_data = np.array([[cars, buses, bikes, hour, day_of_week, is_holiday, weather_index]])
                prediction = model.predict(input_data)[0]
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Bashorat natijasi", f"{prediction}")
                col_m2.success("Model muvaffaqiyatli hisobladi!")
                st.balloons()
            except Exception as e:
                st.error(f"Xatolik: {e}")
        else:
            st.warning("Model fayli ('model (2).pkl') topilmadi.")

with tab2:
    st.subheader("Transport turlari bo'yicha diagramma")
    chart_data = pd.DataFrame({
        'Tur': ['Avtomobil', 'Avtobus', 'Mototsikl'],
        'Soni': [cars, buses, bikes]
    })
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Tur', y='Soni', data=chart_data, palette='viridis', ax=ax1)
    st.pyplot(fig1)

with tab3:
    st.subheader("Polinomial Regressiya va Trend")
    # Vaqt oralig'ini tanlash
    time_start, time_end = st.select_slider("Tahlil uchun vaqt oralig'ini tanlang", options=range(25), value=(6, 22))
    
    # Regressiya trendini simulyatsiya qilish (Polinomial trend line)
    x_range = np.linspace(time_start, time_end, 100)
    y_trend = -0.1 * (x_range - 15)**2 + 50 # Parabolik trend (misol)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(x_range, y_trend, label="Trend chizig'i (Polynomial)", color='red', linewidth=2)
    ax2.scatter([hour], [40], color='blue', s=200, label="Siz tanlagan vaqt") # Hozirgi nuqta
    ax2.set_xlabel("Vaqt (Soat)")
    ax2.set_ylabel("Trafik zichligi")
    ax2.legend()
    ax2.grid(True, linestyle='--')
    st.pyplot(fig2)
    st.info(f"Tanlangan {time_start}:00 dan {time_end}:00 gacha bo'lgan vaqt oralig'idagi regressiya tahlili.")

st.divider()
st.caption("© 2024 Barcha huquqlar himoyalangan. SamDU, Informatika fakulteti.")
