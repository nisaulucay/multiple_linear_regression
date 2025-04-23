import streamlit as st
import joblib
import numpy as np
  
st.title("Maaş tahmin uygulaması")

st.write("Lütfen deneyiminizi giriniz:")
#kullanıcıdan deneyim bilgisi alınır
deneyim = st.number_input("deneyim:",min_value=0, max_value=200, value=1, step=1)

st.write("Lütfen yaşınızı giriniz:")
yas = st.number_input("yas:",min_value=0, max_value=200, value=1, step=1)

if st.button("Hesapla"):
    #model yükleme
    model= joblib.load('multiple_linear_model.pkl')
  
    #tahmin yapılır
    tahmin = model.predict(np.array([[deneyim,yas]]))

    #sonuç
    st.success(f"Tahmini maaş: {float (tahmin[0]):,.2f} TL ")