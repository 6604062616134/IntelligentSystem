import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os
from pathlib import Path
import tempfile



    
MODEL_URL = 'https://drive.usercontent.google.com/download?id=1OzYA4YxjS781ssi_3yQNYiNAgRW4OD2d&export=download&authuser=0&confirm=t&uuid=9ab79fa9-a4e5-4d88-a557-5092d1514473&at=AEz70l6Tqu-HU-J_n_kIWd8Q5J8c:1741102825604'


base = Path(__file__).parent.parent / "NN"
model = base / "model.h5"




if not model.exists():
    os.makedirs(base, exist_ok=True)
    gdown.download(MODEL_URL, str(model), quiet=False)

file_path = Path(__file__).parent.parent / "data NN"
file_cat = file_path / "cat_115.wav"
file_dog = file_path / "dog_barking_46.wav"


# ฟังก์ชันโหลดเสียงและแปลงเป็น Mel Spectrogram
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        if y is None or len(y) == 0:
            raise ValueError(f"ไฟล์ {audio_file} ไม่มีข้อมูลเสียง")

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดไฟล์ {audio_file}: {e}")
        return None
    
model = load_model(model, compile=False)


   


def modele():
    
    st.write("กรุณาเลือกประเภทเสียงที่ต้องการทดสอบ:")

    # ให้ผู้ใช้เลือกประเภทของเสียง
    audio_option = st.selectbox(
        "เลือกประเภทเสียง:",
        ["Cat", "Dog", "เลือกไฟล์ของคุณเอง"]
    )

    audio_path = None  # กำหนดค่าเริ่มต้นให้กับ audio_path

    if audio_option == "Cat":
        audio_path = str(file_cat)
    elif audio_option == "Dog":
        audio_path = str(file_dog)
    elif audio_option == "เลือกไฟล์ของคุณเอง":
        uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียงของคุณเอง", type=["wav", "mp3", "mp4"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            audio_path = temp_file.name

    if not audio_path or not os.path.exists(audio_path):
        st.warning("กรุณาเลือกประเภทเสียงหรืออัพโหลดไฟล์เสียงให้ถูกต้อง")
        return

    # ดึง features จากไฟล์เสียง
    mel_spec = extract_features(audio_path)
    if mel_spec is None:
        return  # หยุดถ้ามีข้อผิดพลาดในการโหลดไฟล์เสียง

    # ปรับขนาดของ Mel Spectrogram
    max_len = 553  # ขนาดที่โมเดลคาดหวัง
    if mel_spec.shape[1] < max_len:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
    else:
        mel_spec = mel_spec[:, :max_len]  # ครอปให้ได้ขนาดที่ต้องการ

    mel_spec = mel_spec[..., np.newaxis]  # เพิ่มมิติให้เหมาะกับโมเดล

    # ตรวจสอบขนาดของข้อมูลที่ป้อนเข้าโมเดล
    if mel_spec.shape != (128, 553, 1):
        st.error(f"ขนาดของข้อมูลที่ป้อนเข้าโมเดลไม่ถูกต้อง: {mel_spec.shape}")
        return

    # ทำนายเสียง
    try:
        prediction = model.predict(np.expand_dims(mel_spec, axis=0))[0][0]
        cat_prob = prediction * 100
        dog_prob = (1 - prediction) * 100

        if cat_prob > dog_prob:
            st.image("https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg", caption="แมว", use_container_width=True)  # แสดงรูปแมว
        else:
            st.image("https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*", caption="หมา", use_container_width=True)  # แสดงรูปหมา


    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนายเสียง: {str(e)}")

    # ลบไฟล์ชั่วคราวหลังใช้งานเสร็จ
    if "temp_file" in locals() and os.path.exists(temp_file.name):  # Use temp_file.name here
        os.unlink(temp_file.name)
    if "audio_path" in locals() and audio_path.startswith("/tmp") and os.path.exists(audio_path):
        os.unlink(audio_path)



def Neural_Network_Model():
    st.title("Neural Network Model")
    st.title("การทำนายเสียงแมวและหมา")
    modele()
    