
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Model va encoderni yuklash
model = pickle.load(open("model(2).pkl", "rb"))
le_sit = pickle.load(open("encoder.pkl", "rb"))

st.set_page_config(page_title="Trafik Bashorati", page_icon="🚦")
st.title("🚦 Aqlli Trafik Bashorat Tizimi")

# Kirish maydonlari
st.sidebar.header("Ma'lumotlarni kiriting")
hour = st.sidebar.slider("Soat", 0, 23, 12)
date = st.sidebar.slider("Sana", 1, 31, 15)
day = st.sidebar.selectbox("Hafta kuni", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Kunni raqamga o'girish (III-bobdagi mantiq asosida)
day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}

car = st.number_input("Avtomobillar soni", 0, 1000, 50)
bike = st.number_input("Mototsikllar soni", 0, 500, 10)
bus = st.number_input("Avtobuslar soni", 0, 100, 5)
truck = st.number_input("Yuk mashinalari soni", 0, 100, 5)

if st.button("🔍 Bashorat qilish"):
    input_data = np.array([[hour, date, day_map[day], car, bike, bus, truck]])
    prediction = model(2).predict(input_data)
    result = le_sit.inverse_transform(prediction)[0]
    
    st.success(f"🚦 Yo'ldagi holat: **{result}**")
