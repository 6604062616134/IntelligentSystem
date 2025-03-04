import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os
from pathlib import Path
import tempfile

# ลิงก์ดาวน์โหลดโมเดล
MODEL_URL = 'https://drive.google.com/uc?id=1OzYA4YxjS781ssi_3yQNYiNAgRW4OD2d'

# กำหนดพาธสำหรับโมเดล
base = Path(__file__).parent / "NN"
model_path = base / "model.h5"

# ตรวจสอบว่ามีโมเดลอยู่หรือไม่ ถ้าไม่มีให้ดาวน์โหลด
if not model_path.exists():
    os.makedirs(base, exist_ok=True)
    gdown.download(MODEL_URL, str(model_path), quiet=False)

# โหลดโมเดล
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
    st.stop()

# ฟังก์ชันโหลดเสียงและแปลงเป็น Mel Spectrogram
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ {audio_file}: {e}")
        return None

# ฟังก์ชันทำนายเสียง
def modele():
    st.title("ทดสอบการทำนายเสียงแมวและหมา")
    
    audio_option = st.selectbox(
        "เลือกประเภทเสียง:",
        ["Cat", "Dog", "เลือกไฟล์ของคุณเอง"]
    )
    
    audio_path = None  # กำหนดค่าเริ่มต้น
    
    if audio_option == "เลือกไฟล์ของคุณเอง":
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียงของคุณเอง", type=["wav", "mp3", "mp4"])
        if uploaded_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            audio_path = temp_file.name
    
    if not audio_path:
        st.warning("กรุณาเลือกประเภทเสียงหรืออัพโหลดไฟล์เสียง")
        return
    
    mel_spec = extract_features(audio_path)
    if mel_spec is None:
        return
    
    max_len = 553
    if mel_spec.shape[1] < max_len:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
    else:
        mel_spec = mel_spec[:, :max_len]
    
    mel_spec = mel_spec[..., np.newaxis]
    
    try:
        prediction = model.predict(np.expand_dims(mel_spec, axis=0))[0][0]
        cat_prob = prediction * 100
        dog_prob = (1 - prediction) * 100
        
        if cat_prob > dog_prob:
            st.image("https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg", caption="แมว", use_container_width=True)
        else:
            st.image("https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*", caption="หมา", use_container_width=True)
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนายเสียง: {str(e)}")
    
    if os.path.exists(audio_path):
        os.unlink(audio_path)

# เรียกใช้งาน Streamlit
def Neural_Network_Model():
    modele()