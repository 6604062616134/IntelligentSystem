import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st
import numpy as np

# Function to calculate different metrics
def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    return mae, mse, r2, mape, accuracy

# Main function to load and process data
def machine_learning():
    try:
        # Set file paths for models and data
        base_path = Path(__file__).parent.parent / "machine learning"
        model_1 = joblib.load(base_path / "rf_temp.pkl")
        model_2 = joblib.load(base_path / "xgb_temp.pkl")
        model_3 = joblib.load(base_path / "svm_temp.pkl")
        ensemble_model = joblib.load(base_path / "ensemble_model_temp.pkl")
        scaler = joblib.load(base_path / "scaler.pkl")
        file_path = Path(__file__).parent.parent / "data"
        file_csv = file_path / "weather_100_years_extended (1).csv"

        # Load data from CSV
        df = pd.read_csv(file_csv)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)

        # Clean up column names and reset index
        df.columns = df.columns.str.strip()
        df.reset_index(drop=True, inplace=True)
        df.index += 1  # Start index from 1

        # Show loaded data in Streamlit
        st.markdown('<div class="highlight"><br>ข้อมูลในไฟล์ที่อัปโหลด</div>', unsafe_allow_html=True)
        st.dataframe(df)

        # Define required columns
        required_columns = ['date', 'country', 'city', 'temperature (°C)', 'humidity (%)', 'wind_speed (km/h)']
        
        # ตรวจสอบคอลัมน์ที่ต้องการ
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # ลบคอลัมน์ที่ไม่จำเป็น
        X = df.drop(columns=['date'])

        # เติมค่าที่หายไปในคอลัมน์ 'temperature (°C)' โดยใช้ฟังก์ชัน forward fill
        y_temp = df['temperature (°C)'].ffill()

        # แปลงข้อมูล 'country' และ 'city' เป็นตัวเลข (One-Hot Encoding)
        df = pd.get_dummies(df, columns=['country', 'city'], drop_first=True)

        # แปลงค่า True/False ให้เป็น 0/1 (ถ้ามี)
        bool_columns = df.select_dtypes(include=['bool']).columns
        df[bool_columns] = df[bool_columns].astype(int)

        # แสดงข้อมูลที่แปลงแล้ว
        st.write("ข้อมูลที่แปลงแล้ว:")
        st.dataframe(df.head())

        # แยกข้อมูลตัวเลขและข้อมูลหมวดหมู่
        
        numeric_cols = X.select_dtypes(include=['number']).columns  # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
        categorical_cols = X.select_dtypes(exclude=['number']).columns  # เลือกเฉพาะคอลัมน์ที่เป็นข้อความ

        # ตรวจสอบว่ามีคอลัมน์ตัวเลขหรือไม่
        if len(numeric_cols) == 0:
            raise ValueError("ไม่มีคอลัมน์ตัวเลขเหลืออยู่หลังการแปลงข้อมูล")

        # เติมค่าที่หายไปในข้อมูลตัวเลขด้วยค่าเฉลี่ย (mean)
        num_imputer = SimpleImputer(strategy='mean')
        X_numeric_imputed = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

        # เติมค่าที่หายไปในข้อมูลหมวดหมู่ด้วยค่าที่พบบ่อยที่สุด (most_frequent) ถ้ามีคอลัมน์หมวดหมู่
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_categorical_imputed = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)
        else:
            X_categorical_imputed = pd.DataFrame()  # ถ้าไม่มีคอลัมน์หมวดหมู่ ให้สร้าง DataFrame ว่าง

        # รวมข้อมูลทั้งสองประเภทกลับมา
        X_imputed = pd.concat([X_numeric_imputed, X_categorical_imputed], axis=1)

        # ตรวจสอบคอลัมน์ที่เติมค่ากลับมา
        missing_values = X_imputed.isnull().sum()
        st.write("ค่าที่หายไปหลังการเติมค่ากลับมา:")
        st.write(missing_values)

        # แปลงข้อมูลตัวเลขโดยใช้ StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(X_imputed[numeric_cols]), columns=numeric_cols)

        # ทำนายผลด้วยโมเดลที่โหลดมา
        pred_1 = model_1.predict(X_scaled)
        pred_2 = model_2.predict(X_scaled)
        pred_3 = model_3.predict(X_scaled)
        pred_ensemble = ensemble_model.predict(X_scaled)

        # คำนวณค่าประเมินผล
        mae_1, mse_1, r2_1, mape_1, accuracy_1 = calculate_metrics(y_temp, pred_1)
        mae_2, mse_2, r2_2, mape_2, accuracy_2 = calculate_metrics(y_temp, pred_2)
        mae_3, mse_3, r2_3, mape_3, accuracy_3 = calculate_metrics(y_temp, pred_3)
        mae_ensemble, mse_ensemble, r2_ensemble, mape_ensemble, accuracy_ensemble = calculate_metrics(y_temp, pred_ensemble)

        # สร้าง DataFrame แสดงผล
        result_data = {
            'Model': ['RF', 'xgb', 'SVR', 'Ensemble'],
            'MAE': [mae_1, mae_2, mae_3, mae_ensemble],
            'MSE': [mse_1, mse_2, mse_3, mse_ensemble],
            'R2': [r2_1, r2_2, r2_3, r2_ensemble],
            'MAPE (%)': [mape_1, mape_2, mape_3, mape_ensemble],
            'Prediction Accuracy (%)': [accuracy_1, accuracy_2, accuracy_3, accuracy_ensemble]
        }

        results_df = pd.DataFrame(result_data)

        # แสดงผลการประเมิน
        st.markdown('<br><div class="big-font">ผลการประเมินประสิทธิภาพของโมเดล</div>', unsafe_allow_html=True)
        st.dataframe(results_df.style.format({'MAE': '{:.4f}', 'MSE': '{:.4f}', 'R2': '{:.4f}', 'MAPE (%)': '{:.4f}', 'Prediction Accuracy (%)': '{:.4f}'}))

        # --- Create Density Plot --- #
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(y_temp, label='Actual', ax=ax, color='black', linestyle='--')
        sns.kdeplot(pred_1, label='RF', ax=ax, color='blue')
        sns.kdeplot(pred_2, label='xgb', ax=ax, color='green')
        sns.kdeplot(pred_3, label='SVR', ax=ax, color='red')
        sns.kdeplot(pred_ensemble, label='Ensemble', ax=ax, color='orange', linestyle='-.')

        ax.set_title('Density Plot of Model Predictions vs Actual Values')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Density')
        ax.legend()

        # Display the plot
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("ไม่พบไฟล์ CSV ในโฟลเดอร์")
    except ValueError as e:
        st.error(f"ข้อผิดพลาด: {str(e)}")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {str(e)}")
