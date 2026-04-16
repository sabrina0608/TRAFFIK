import streamlit as st
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Any

# 1. Logging sozlamalari (Xatolarni kuzatish uchun professional yondashuv)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInterface:
    """Model bilan ishlash uchun maxsus interfeys klassi."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[Any] = None
        self.base_path = Path(__file__).parent

    @st.cache_resource
    def load_model(_self) -> bool:
        """Modelni xotiraga (cache) yuklash."""
        model_path = _self.base_path / _self.model_name
        
        if not model_path.exists():
            logger.error(f"Model fayli topilmadi: {model_path}")
            return False
        
        try:
            with open(model_path, "rb") as f:
                _self.model = pickle.load(f)
            logger.info("Model muvaffaqiyatli yuklandi.")
            return True
        except (pickle.UnpicklingError, IOError) as e:
            logger.error(f"Modelni o'qishda xatolik: {e}")
            return False

# 2. UI/UX Dizayn va Sahifa sozlamalari
def setup_ui():
    st.set_page_config(
        page_title="AI Prediction Pro",
        page_icon="🤖",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_ui()
    
    st.title("🤖 Professional AI Bashorat Tizimi")
    st.sidebar.header("Sozlamalar")
    
    # Modelni inisializatsiya qilish
    predictor = ModelInterface("model(2).pkl")
    
    with st.spinner("Model yuklanmoqda..."):
        if not predictor.load_model():
            st.error("❌ Tizimda nosozlik: Model fayli topilmadi yoki shikastlangan.")
            st.info("Eslatma: 'model(2).pkl' fayli app.py bilan bir xil papkada ekanligini tekshiring.")
            return

    # 3. Ma'lumotlarni kiritish qismi (Layout)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Ma'lumotlarni kiriting")
        # Misol tariqasida inputlar
        feature_1 = st.number_input("Xususiyat 1", help="Bu yerga birinchi qiymatni kiriting")
        feature_2 = st.number_input("Xususiyat 2", help="Bu yerga ikkinchi qiymatni kiriting")

    with col2:
        st.subheader("📊 Natija")
        if st.button("Bashoratni hisoblash"):
            try:
                # Ma'lumotni modelga tayyorlash
                input_array = np.array([[feature_1, feature_2]])
                prediction = predictor.model.predict(input_array)
                
                # Natijani chiroyli ko'rsatish
                st.success(f"Bashorat qilingan natija: **{prediction[0]}**")
                st.metric(label="Aniqroq natija", value=f"{prediction[0]:.2f}")
            except Exception as e:
                st.error(f"Hisoblashda xatolik yuz berdi: {e}")

if __name__ == "__main__":
    main()
