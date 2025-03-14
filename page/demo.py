import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from pathlib import Path
from NN import modele

def demo():
    # กำหนดตำแหน่งไฟล์โมเดลและข้อมูล
    base_path = Path(__file__).parent.parent / "machine learning"
    
    # โหลดโมเดล
    model_1 = joblib.load(base_path / "rf_temp.pkl")  # Random Forest Model
    model_2 = joblib.load(base_path / "xgb_temp.pkl")  # XGBoost Model
    model_3 = joblib.load(base_path / "svm_temp.pkl")  # SVM Model
    ensemble_model = joblib.load(base_path / "ensemble_model_temp.pkl")  # Ensemble Model
    
    # โหลด Scaler
    scaler = joblib.load(base_path / "scaler.pkl")

    st.title("ทำนายอุณหภูมิจากข้อมูลสภาพอากาศ")

    # เลือกประเทศ
    country = st.selectbox("เลือกประเทศ", ["Thailand", "UK", "USA", "South Korea", "Japan", "France", "China"])

    # เลือกเมือง (กำหนดตัวเลือกเมืองตามประเทศที่เลือก)
    city_options = {
        "Thailand": ["Bangkok"],
        "UK": ["London"],
        "USA": ["New York"],
        "South Korea": ["Seoul"],
        "Japan": ["Tokyo"],
        "France": ["Paris"],
        "China": ["Beijing"]
    }
    city = st.selectbox("เลือกเมือง", city_options[country])

    # กรอกข้อมูลที่จำเป็น
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0,value=0.00)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0,value=0.00)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0,value=0.00)

    
    input_data = pd.DataFrame([[temperature, humidity, wind_speed]], 
                              columns=['temperature (°C)', 'humidity (%)', 'wind_speed (km/h)'])

    # เติมค่าที่หายไปด้วยค่าเฉลี่ย
    imputer = SimpleImputer(strategy='mean')
    input_data = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)

    # ทำการ scaling ข้อมูล
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    # เลือกโมเดลที่ใช้ในการทำนาย
    model_choice = st.selectbox("เลือกโมเดลที่ใช้", ["Random Forest", "XGBoost", "SVM", "Ensemble"])

    # ทำนายผลลัพธ์
    if model_choice == "Random Forest":
        pred = model_1.predict(input_data_scaled)
    elif model_choice == "XGBoost":
        pred = model_2.predict(input_data_scaled)
    elif model_choice == "SVM":
        pred = model_3.predict(input_data_scaled)
    elif model_choice == "Ensemble":
        pred_rf = model_1.predict(input_data_scaled)
        pred_xgb = model_2.predict(input_data_scaled)
        pred_svm = model_3.predict(input_data_scaled)
        pred = (pred_rf + pred_xgb + pred_svm) / 3  # ค่าเฉลี่ยจากทุกโมเดล

    # แสดงผลลัพธ์
    st.write(f"อุณหภูมิที่ทำนายสำหรับ  {city}, {country}: {pred[0]:.2f} °C" )
    st.title("Neural Network Model")
    modele()

