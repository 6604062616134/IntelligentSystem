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
    file_path = Path(__file__).parent.parent / "data NN"
    file_cat = file_path / "cat_115.wav"
    file_dog = file_path / "dog_barking_46.wav"
    st.title("ทดสอบการทำนายเสียงแมวและหมา")
    
    audio_option = st.selectbox(
        "เลือกประเภทเสียง:",
        ["Cat", "Dog", "เลือกไฟล์ของคุณเอง"]
    )
    
    audio_path = None  # กำหนดค่าเริ่มต้น
    
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
            st.image("https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg", caption="แมว")
        else:
            st.image("https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*", caption="หมา")
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนายเสียง: {str(e)}")
    
    if "temp_file" in locals() and os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
    if "audio_path" in locals() and audio_path.startswith("/tmp") and os.path.exists(audio_path):
        os.unlink(audio_path)

# เรียกใช้งาน Streamlit
def Neural_Network_Model():
    st.title("Convolutional Neural Networks (CNN) คืออะไร และเกี่ยวข้องกับการทำนายเสียงแมวและหมาอย่างไร?")
       #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)
    st.image("https://www.cell.com/cms/10.1016/j.tins.2022.12.008/asset/a7b7e5a2-485c-4941-8efc-14b39438f31c/main.assets/gr1_lrg.jpg", caption="รูปจาก https://www.cell.com/trends/neurosciences/fulltext/S0166-2236%2822%2900262-4")
   
    st.markdown('<div class="text_indent">ในยุคที่ปัญญาประดิษฐ์ (AI) ได้รับการพัฒนาอย่างก้าวกระโดด หนึ่งในโมเดลที่มีบทบาทสำคัญอย่างมากในด้านการประมวลผลภาพและเสียงคือ Convolutional Neural Networks (CNN) ซึ่งถูกออกแบบมาเพื่อให้สามารถเรียนรู้คุณลักษณะที่ซับซ้อนของข้อมูลรูปภาพและสัญญาณเสียงได้อย่างมีประสิทธิภาพCNN เป็นหนึ่งในประเภทของโครงข่ายประสาทเทียมที่ได้รับความนิยม โดยเฉพาะอย่างยิ่งในการจำแนกภาพ วัตถุ และแม้กระทั่งการวิเคราะห์เสียง มันถูกนำมาใช้ในงานที่ต้องการการเรียนรู้เชิงลึก (Deep Learning) เพื่อแยกแยะและวิเคราะห์รูปแบบที่ซับซ้อนของข้อมูลเสียงเช่นกัน </div><br>', unsafe_allow_html=True)
   
    st.markdown('<div class="big-font"> Convolutional Neural Networks (CNN) คืออะไร?</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">CNN เป็นโครงข่ายประสาทเทียมชนิดหนึ่งที่ได้รับการออกแบบมาเพื่อประมวลผลข้อมูลที่มีโครงสร้างเป็นกริด เช่น รูปภาพ หรือ สัญญาณเสียง ซึ่ง CNN ประกอบด้วยหลายชั้น (Layers) ที่ทำงานร่วมกัน ได้แก่ </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Convolutional Layer: ทำหน้าที่สกัดคุณลักษณะจากข้อมูลนำเข้าโดยใช้ฟิลเตอร์หรือเคอร์เนล (Kernel) เพื่อสร้างฟีเจอร์แมป (Feature Map) ที่เป็นตัวแทนของข้อมูลต้นฉบับ </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Pooling Layer: ลดขนาดของฟีเจอร์แมปลง เพื่อช่วยลดจำนวนพารามิเตอร์และเพิ่มความทนทานต่อการเปลี่ยนแปลงของข้อมูล </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> Fully Connected Layer: นำฟีเจอร์ที่สกัดมาได้ไปประมวลผลในรูปแบบของโครงข่ายประสาทเทียมดั้งเดิมเพื่อทำการจำแนกประเภท</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> Activation Function: ใช้ฟังก์ชันเชิงไม่เชิงเส้น เช่น ReLU, Sigmoid หรือ Softmax เพื่อเพิ่มความสามารถในการเรียนรู้ของเครือข่าย</div><br>', unsafe_allow_html=True)

    st.markdown('<div class="big-font"> CNN กับการทำนายเสียงแมวและหมา</div>', unsafe_allow_html=True)
    st.image("https://cdn.pixabay.com/photo/2018/10/01/09/21/pets-3715733_960_720.jpg")
    st.markdown('<div class="text_indent"> โดยปกติแล้ว CNN ถูกใช้ในงานที่เกี่ยวข้องกับการจำแนกรูปภาพเป็นหลัก อย่างไรก็ตาม CNN สามารถนำมาใช้กับข้อมูลเสียงได้เช่นกันผ่านการแปลงข้อมูลเสียงเป็นรูปแบบของ Spectrogram ซึ่งเป็นการแสดงสัญญาณเสียงในรูปแบบของภาพ โดย Spectrogram แสดงค่าความถี่ตามแนวตั้ง และเวลาในแนวนอน โดยมีระดับความเข้มของสีแสดงถึงพลังงานของเสียงที่ตำแหน่งนั้น ๆ</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="highlight"> Mel Spectrogram ทำงานอย่างไร</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Mel Spectrogram เป็นการแปลงสัญญาณเสียงให้อยู่ในรูปของภาพที่แสดงระดับพลังงานของความถี่ในช่วงเวลาหนึ่ง ๆ แต่แตกต่างจาก Spectrogram ทั่วไปตรงที่ Mel Spectrogram ใช้มาตราส่วนความถี่ที่ใกล้เคียงกับการได้ยินของมนุษย์ โดยอ้างอิงจาก Mel Scale ซึ่งเป็นการแปลงค่าความถี่เชิงเส้นให้มีความคล้ายกับการรับรู้เสียงของมนุษย์มากขึ้น </div><br>', unsafe_allow_html=True)
    st.image("https://ars.els-cdn.com/content/image/1-s2.0-S0957417423000520-gr3.jpg", caption="รูปจาก https://www.sciencedirect.com/science/article/pii/S0957417423000520")
    
    
    
    st.markdown('<div class="big-font"> การเตรียมชุดข้อมูลเสียงหมาและแมว</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> สำหรับการศึกษานี้ เราใช้ชุดข้อมูลเสียงของสุนัขและแมวที่ได้รวบรวมมาจากแหล่งข้อมูลโอเพ่นซอร์ส โดยชุดข้อมูลนี้แบ่งออกเป็น 2 คลาส ได้แก่</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• Dog (สุนัข) - ไฟล์เสียงที่ประกอบไปด้วยเสียงเห่าหรือเสียงร้องของสุนัข</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• Cat (แมว) - ไฟล์เสียงที่เป็นเสียงร้องของแมว เช่น "เมี้ยว"</div><br>', unsafe_allow_html=True)
   
    st.markdown('<div class="text_indent">ในแต่ละคลาส เรามีไฟล์เสียงรวมกันประมาณ 1,000 ไฟล์ โดยแต่ละไฟล์มีความยาวที่แตกต่างกัน  </div><br>', unsafe_allow_html=True)
    
    st.markdown('<br><div class="highlight"> การแปลงเสียงเป็น Mel Spectrogram</div>', unsafe_allow_html=True)
    code = '''
import librosa
import numpy as np

def extract_features(audio_file):
    try:
        # โหลดไฟล์เสียง
        y, sr = librosa.load(audio_file, sr=None)
        
        # ตรวจสอบว่าไฟล์เสียงมีข้อมูลหรือไม่
        if y is None or len(y) == 0:
            raise ValueError(f"ไฟล์ {audio_file} ไม่มีข้อมูลเสียง")

        # คำนวณ Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        
        # แปลงพลังงานของ Mel Spectrogram ให้อยู่ในรูปแบบ Log Scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดไฟล์ {audio_file}: {e}")
        return None

        '''
    st.code(code, language='python')   
    st.markdown('<div class="text_indent"> เพื่อให้โมเดลสามารถเรียนรู้ลักษณะของเสียงได้ดีขึ้น เราต้องแปลงข้อมูลเสียงดิบให้อยู่ในรูปแบบของ Mel Spectrogram ซึ่งเป็นการแสดงผลความเข้มของพลังงานเสียงในแต่ละช่วงความถี่ตามมาตรฐาน Mel-scale ที่ใกล้เคียงกับการรับรู้เสียงของมนุษย์ ฟังก์ชัน extract_features ทำหน้าที่นี้โดยใช้ไลบรารี librosa</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">1. โหลดไฟล์เสียงใช้ librosa.load(audio_file, sr=None) เพื่อโหลดไฟล์เสียงเข้ามา โดยกำหนดให้ sr=None เพื่อใช้ค่า Sampling Rate ดั้งเดิมของไฟล์เสียง</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">2. ตรวจสอบข้อมูลเสียงหากไฟล์เสียงไม่มีข้อมูล (เช่น ไฟล์เสียหายหรือเป็นไฟล์ว่างเปล่า) ระบบจะคืนค่า Error และแจ้งเตือนผู้ใช้ </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">3. คำนวณ Mel Spectrogramใช้ librosa.feature.melspectrogram เพื่อแปลงสัญญาณเสียงให้อยู่ในรูปแบบของ Mel Spectrogram โดยกำหนด n_mels=128 → ใช้ 128 เมลฟิลเตอร์ , fmax=8000 → กำหนดค่าความถี่สูงสุดที่ 8000 Hz เพื่อให้ได้ช่วงความถี่ที่เหมาะสม </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">4. แปลงเป็น Log Scaleค่า Mel Spectrogram ที่ได้จะถูกแปลงให้อยู่ในหน่วยเดซิเบล (dB) ด้วย librosa.power_to_db(mel_spec, ref=np.max) เพื่อช่วยให้โมเดลสามารถเรียนรู้ค่าความเข้มเสียงได้ดีขึ้น</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">5. การจัดการข้อผิดพลาดหากมีข้อผิดพลาด เช่น ไฟล์เสียงเสียหายหรือโหลดไม่สำเร็จ ฟังก์ชันจะพิมพ์ข้อความแจ้งเตือนและคืนค่า None</div><br>', unsafe_allow_html=True)

    st.markdown('<br><div class="highlight">ตรวจสอบไดเรกทอรี </div>', unsafe_allow_html=True)
    code = '''
dog_dir = "/content/drive/MyDrive/dog_wav"
cat_dir = "/content/drive/MyDrive/cat_wav"

# ตรวจสอบว่าไดเรกทอรีมีอยู่
if os.path.exists(dog_dir):
    print(f"Dog directory exists: {dog_dir}")
else:
    print(f"Dog directory does not exist: {dog_dir}")

if os.path.exists(cat_dir):
    print(f"Cat directory exists: {cat_dir}")
else:
    print(f"Cat directory does not exist: {cat_dir}")

        '''
    st.code(code, language='python')   
    st.markdown('<div class="text_indent"> ตรวจสอบว่าโฟลเดอร์ที่เก็บไฟล์เสียงของหมา (dog_wav) และแมว (cat_wav) มีอยู่หรือไม่</div><br>', unsafe_allow_html=True)
    

    st.markdown('<br><div class="highlight">ฟังก์ชันดึงคุณลักษณะจากไฟล์เสียง </div>', unsafe_allow_html=True)
    code = '''
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # โหลดไฟล์เสียง
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)  # แปลงเป็น Mel Spectrogram
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # ทำให้เป็น dB scale
        return mel_spec_db
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

        '''
    st.code(code, language='python')   
    st.markdown('<div class="text_indent"> โหลดไฟล์เสียงแล้วแปลงเป็น Mel Spectrogram</div><br>', unsafe_allow_html=True)
    
    
    st.markdown('<br><div class="highlight"> โหลดไฟล์เสียงจากไดเรกทอรี</div>', unsafe_allow_html=True)
    code = '''
# อ่านไฟล์ทั้งหมดจากไดเรกทอรี
dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.endswith('.wav')]
cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.wav')]

        '''
    st.code(code, language='python')   
    st.markdown('<div class="text_indent"> ค้นหาไฟล์ .wav ทั้งหมดในโฟลเดอร์ของหมา (dog_wav) และแมว (cat_wav)</div><br>', unsafe_allow_html=True)
   

    st.markdown('<br><div class="highlight"> ดึงฟีเจอร์และสร้างป้ายกำกับ</div>', unsafe_allow_html=True)
    code = '''
    # รวมไฟล์เสียงและแท็ก
features, labels = [], []

# สำหรับไฟล์เสียงหมา
for file in dog_files:
    mel_spec = extract_features(file)
    if mel_spec is not None:
        features.append(mel_spec)
        labels.append(0)  # 0 = dog

# สำหรับไฟล์เสียงแมว
for file in cat_files:
    mel_spec = extract_features(file)
    if mel_spec is not None:
        features.append(mel_spec)
        labels.append(1)  # 1 = cat

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• วนลูปอ่านไฟล์ .wav ของหมาและแมวและทำการ  </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• ดึงคุณลักษณะจากเสียงด้วย extract_features()</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• ถ้าดึงข้อมูลสำเร็จ ให้เก็บ Mel Spectrogram ลงใน features</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• เพิ่มป้ายกำกับ (0 = หมา, 1 = แมว) ลงใน labels</div><br>', unsafe_allow_html=True)

  
    st.markdown('<br><div class="highlight"> ตรวจสอบว่ามีข้อมูลหรือไม่</div>', unsafe_allow_html=True)
    code = '''
if not features:
    raise ValueError("ไม่สามารถโหลดไฟล์เสียงได้ ตรวจสอบไฟล์ต้นฉบับอีกครั้ง")

        '''
    st.code(code, language='python')   
    st.markdown('<div class="text_indent">ตรวจสอบว่ามีข้อมูลเสียงถูกโหลดเข้ามาหรือไม่ ถ้าไม่มีไฟล์เสียงที่ถูกโหลดสำเร็จ จะเกิด ValueError พร้อมข้อความแจ้งเตือนให้ตรวจสอบไฟล์อีกครั้ง </div><br>', unsafe_allow_html=True)
   




    st.markdown('<br><div class="highlight">ปรับขนาดของฟีเจอร์ให้เท่ากัน </div>', unsafe_allow_html=True)
    code = '''
max_length = max([feature.shape[1] for feature in features])  # หาขนาดสูงสุด
features_padded = [np.pad(feature, ((0, 0), (0, max_length - feature.shape[1])), mode='constant') for feature in features]

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• Mel Spectrogram แต่ละไฟล์อาจมีความยาวต่างกัน ขึ้นอยู่กับความยาวของเสียง</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• หาความกว้าง (จำนวนเฟรม) ที่ยาวที่สุดของ Mel Spectrogram จาก features</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• ใช้ np.pad() เพื่อเติม 0 ที่ส่วนท้ายของสเปกโทรแกรมที่มีขนาดเล็กกว่า เพื่อให้ทุกรูปมีขนาดเดียวกัน</div><br>', unsafe_allow_html=True)
   



    st.markdown('<br><div class="highlight"> แปลงเป็นอาร์เรย์ NumPy</div>', unsafe_allow_html=True)
    code = '''
X = np.array(features_padded)
y = np.array(labels)

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• แปลง features_padded และ labels ให้อยู่ในรูปแบบ numpy array เพื่อให้ใช้กับโมเดลได้ง่าย</div><br>', unsafe_allow_html=True)
    

    st.markdown('<br><div class="highlight"> แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ</div>', unsafe_allow_html=True)
    code = '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• แบ่งข้อมูลเป็น ชุดฝึก (train) และ ชุดทดสอบ (test)</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• test_size=0.2 หมายถึงใช้ 20% ของข้อมูลเป็นชุดทดสอบ และ 80% เป็นชุดฝึก</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• random_state=42 กำหนดค่า seed เพื่อให้การสุ่มแบ่งข้อมูลให้ได้ผลลัพธ์เดิมทุกครั้ง</div><br>', unsafe_allow_html=True)
    

    st.markdown('<br><div class="highlight"> เพิ่มมิติให้กับข้อมูล</div>', unsafe_allow_html=True)
    code = '''
X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">•Mel Spectrogram มีขนาด (128, max_length), แต่โมเดล CNN ต้องการรูปแบบ 3D (เช่น รูปภาพ)</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• np.newaxis เพิ่มมิติที่ 3 ทำให้ X_train และ X_test มีขนาดเป็น (จำนวนตัวอย่าง, 128, max_length, 1)</div><br>', unsafe_allow_html=True)

    st.markdown('<div class="big-font"> การสร้างโมเดล CNN</div><br>', unsafe_allow_html=True)




    st.markdown('<br><div class="highlight">Input Layer </div>', unsafe_allow_html=True)
    code = '''
layers.InputLayer(shape=(X_train.shape[1], X_train.shape[2], 1))


        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• เลเยอร์นี้ใช้กำหนดขนาดของข้อมูลอินพุตที่เข้าโมเดล (ความสูง, ความกว้าง, จำนวนช่องสี).</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• X_train.shape[1] และ X_train.shape[2] คือขนาดของรูปภาพ (ความสูงและความกว้าง) และ 1 หมายถึงข้อมูลภาพมีแค่ช่องสีเดียว (เช่น ขาว-ดำ หรือ Grayscale</div><br>', unsafe_allow_html=True)

    st.markdown('<br><div class="highlight">onv2D (Convolutional Layer)</div>', unsafe_allow_html=True)
    code = '''
layers.Conv2D(32, (3, 3), activation='relu')

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">•32: จำนวนฟิลเตอร์ที่ใช้ในเลเยอร์นี้ ซึ่งหมายความว่าโมเดลจะสร้าง 32 ฟีเจอร์แมพจากข้อมูลอินพุต</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• (3, 3): ขนาดของฟิลเตอร์ (3x3) ที่ใช้ในการทำคอนโวลูชัน</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• activation=relu: ฟังก์ชัน activation แบบ ReLU (Rectified Linear Unit) ซึ่งช่วยเพิ่มความสามารถในการเรียนรู้</div><br>', unsafe_allow_html=True)

    st.markdown('<br><div class="highlight">MaxPooling2D </div>', unsafe_allow_html=True)
    code = '''
layers.MaxPooling2D((2, 2))

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• เลเยอร์นี้จะย่อขนาดของข้อมูลโดยใช้การจับคู่พิกเซล 2x2 และเก็บค่าสูงสุดในแต่ละกลุ่มพิกเซล</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• ช่วยลดขนาดของข้อมูลและเพิ่มความทนทานให้กับโมเดลในการจัดการกับการเปลี่ยนแปลงเล็กน้อยของภาพ</div><br>', unsafe_allow_html=True)

    st.markdown('<br><div class="highlight">Conv2D และ MaxPooling2D ซ้ำ</div>', unsafe_allow_html=True)
    code = '''
layers.Conv2D(64, (3, 3), activation='relu')
layers.MaxPooling2D((2, 2))

layers.Conv2D(128, (3, 3), activation='relu')
layers.MaxPooling2D((2, 2))

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• เช่นเดียวกับก่อนหน้านี้ แต่ในที่นี้ใช้ฟิลเตอร์ที่มากขึ้น (64 และ 128) เพื่อเรียนรู้ฟีเจอร์ที่ซับซ้อนขึ้นจากข้อมูลภาพ</div><br>', unsafe_allow_html=True)
   

    st.markdown('<br><div class="highlight">Flatten </div>', unsafe_allow_html=True)
    code = '''
layers.Flatten()

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• การแปลงข้อมูลที่มีหลายมิติ (เช่น 2D ของรูปภาพ) ให้เป็นเวกเตอร์ 1D เพื่อใช้กับเลเยอร์ที่เป็น Dense (เลเยอร์เชื่อมโยง)</div><br>', unsafe_allow_html=True)
 

    st.markdown('<br><div class="highlight">Dense Layers </div>', unsafe_allow_html=True)
    code = '''
layers.Dense(128, activation='relu')
layers.Dense(64, activation='relu')

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• เลเยอร์ที่มีการเชื่อมโยงเต็มรูปแบบ (Fully Connected Layers)</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• เลเยอร์แรกมี 128 นิวรอนและใช้ฟังก์ชัน activation ReLU, ส่วนเลเยอร์ที่สองมี 64 นิวรอนและใช้ ReLU เช่นกัน</div><br>', unsafe_allow_html=True)

    st.markdown('<br><div class="highlight">Output Layer (Dense with Sigmoid) </div>', unsafe_allow_html=True)
    code = '''
layers.Dense(1, activation='sigmoid')

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• เลเยอร์สุดท้ายเป็นเลเยอร์ที่ใช้สำหรับการจำแนก 2 คลาส</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• จำนวนของนิวรอนในเลเยอร์นี้คือ 1 เนื่องจากการจำแนกเป็นคลาสเดียว (เช่น ป่วย/ไม่ป่วย)</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• ฟังก์ชัน activation แบบ sigmoid จะให้ค่าในช่วง [0, 1] ซึ่งจะใช้ในการตัดสินใจว่าอินพุตอยู่ในคลาสใด</div><br>', unsafe_allow_html=True)

    st.markdown('<br><div class="highlight"> การคอมไพล์โมเดล</div>', unsafe_allow_html=True)
    code = '''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• optimizer=adam: ใช้อัลกอริธึม Adam ที่เป็นหนึ่งในออพติไมเซอร์ที่นิยมใช้ในการเรียนรู้</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• loss=binary_crossentropy: ใช้ฟังก์ชันการสูญเสีย binary cross-entropy เนื่องจากเป็นปัญหาการจำแนก 2 คลาส (เช่น ใช่/ไม่ใช่)</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• metrics=[accuracy]: กำหนดให้ใช้ความแม่นยำ (accuracy) เป็นการวัดผลลัพธ์ในการฝึกสอน</div><br>', unsafe_allow_html=True)

    

    st.markdown('<br><div class="highlight"> การฝึกโมเดล</div>', unsafe_allow_html=True)
    code = '''
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

        '''
    st.code(code, language='python')   
    st.markdown('<div class="normal-text">• จำนวนรอบในการฝึก (epochs) ที่โมเดลจะเรียนรู้จากข้อมูล X_train และ y_train. ในที่นี้คือ 100 รอบ. การฝึก 1 รอบหมายถึงการใช้ข้อมูลทั้งหมดในชุดฝึกมาผ่านโมเดล 1 ครั้ง </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• จำนวนรอบการฝึกมากขึ้นช่วยให้โมเดลมีโอกาสเรียนรู้ฟีเจอร์มากขึ้น แต่ถ้ามากเกินไปอาจทำให้โมเดล overfit กับข้อมูลฝึกได้</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• จำนวนตัวอย่างในแต่ละ batch ที่ใช้ในการอัพเดตน้ำหนัก (weights) ของโมเดล. ในที่นี้คือ 16 ตัวอย่างต่อ 1 batch</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• การเลือกขนาด batch size ที่เหมาะสมสามารถช่วยให้การฝึกโมเดลมีประสิทธิภาพมากขึ้น. ขนาดเล็กอาจทำให้โมเดลเรียนรู้เร็วขึ้นแต่มีความผันผวนสูง, ขนาดใหญ่ช่วยให้โมเดลมีความเสถียรมากขึ้น</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• ข้อมูลที่ใช้สำหรับการตรวจสอบ (validation) ขณะฝึกโมเดล. ข้อมูลเหล่านี้จะไม่ถูกใช้ในการฝึกโมเดล แต่จะใช้เพื่อตรวจสอบว่าประสิทธิภาพของโมเดลเป็นอย่างไรในการทำนายข้อมูลที่ไม่เคยเห็นมาก่อน</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• X_test คือข้อมูลทดสอบ (inputs) และ y_test คือค่าผลลัพธ์ที่ต้องการให้โมเดลทำนาย</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">• การใช้ข้อมูล validation ช่วยให้สามารถติดตามว่ามีการ overfitting หรือไม่ (โมเดลเริ่มเรียนรู้แค่ข้อมูลฝึก แต่ไม่สามารถทำงานกับข้อมูลใหม่ได้ดี)</div><br>', unsafe_allow_html=True)

    #st.markdown('<div class="big-font"> </div>', unsafe_allow_html=True)

    #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)
    #st.markdown('<div class="text_indent"> </div><br>', unsafe_allow_html=True)
    
    #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)
    #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)


    modele()