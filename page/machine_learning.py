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
    st.title("การประมวลผลข้อมูลและทำนายอุณหภูมิ")
    st.image("https://www.js100.com/uploads/news/5ebf7569ba30ecff39c70212da078f42.jpg")
    st.markdown('<div class="text_indent"> ในโลกของการวิเคราะห์ข้อมูล การพยากรณ์อุณหภูมิที่แม่นยำถือเป็นหนึ่งในองค์ประกอบที่สำคัญในการวางแผนและตัดสินใจในหลายภาคส่วน เช่น เกษตรกรรม พลังงาน และการท่องเที่ยว ซึ่งการคาดการณ์ที่ถูกต้องสามารถช่วยให้ธุรกิจและกิจกรรมต่างๆ ดำเนินไปได้อย่างมีประสิทธิภาพมากขึ้น โปรเจกต์นี้มุ่งเน้นการพัฒนาระบบพยากรณ์อุณหภูมิ โดยใช้ข้อมูลสภาพอากาศที่สำคัญ ได้แก่ ประเทศ (country), เมือง (city), อุณหภูมิ (temperature °C), ความชื้น (humidity %), และความเร็วลม (wind speed km/h) ข้อมูลเหล่านี้จะถูกนำมาวิเคราะห์เพื่อสร้างแบบจำลองที่สามารถทำนายอุณหภูมิได้อย่างแม่นยำ ซึ่งการเตรียมข้อมูลและการเลือกเทคนิคการวิเคราะห์ที่เหมาะสมเป็นขั้นตอนสำคัญในการสร้างระบบพยากรณ์ที่มีประสิทธิภาพสูงสุด</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> ก่อนที่เราจะเข้าไปในเนื้อหาของการเตรียมข้อมูล (Data Preparation) ในบทความนี้ เราจะมาดูกันว่าแต่ละประเภทของข้อมูลเหล่านี้มีความสำคัญอย่างไรในการพยากรณ์อุณหภูมิและการนำมาวิเคราะห์ในขั้นตอนต่างๆ</div>', unsafe_allow_html=True)
    st.image("https://datanorth.ai/wp-content/uploads/2024/08/data_preperation_header.jpg")
    st.markdown('<div class="text_indent"> เมื่อเราเริ่มต้นการพยากรณ์อุณหภูมิด้วยข้อมูลสภาพอากาศที่สำคัญเหล่านี้ เราจะต้องทำความเข้าใจถึงบทบาทของแต่ละประเภทข้อมูลในกระบวนการวิเคราะห์ ซึ่งจะช่วยให้สามารถพัฒนาระบบที่มีความแม่นยำสูงขึ้นได้ ดังนี้</div><br>', unsafe_allow_html=True)
   
    st.markdown('<div class="highlight">ประเทศ (Country) และ เมือง (City)</div>', unsafe_allow_html=True)
    st.image("https://blog.takemetour.com/wp-content/uploads/2016/02/why-Bangkok-is-the-best-city-view.jpg")
   
    st.markdown('<div class="text_indent"> ข้อมูลในส่วนนี้เป็นข้อมูลพื้นฐานที่มีผลต่อการแปรผันของสภาพอากาศในแต่ละสถานที่ เนื่องจากแต่ละประเทศและเมืองมีลักษณะทางภูมิศาสตร์และภูมิอากาศที่แตกต่างกัน โดยการใช้ข้อมูลประเทศและเมืองสามารถช่วยในการจำแนกและปรับตัวแบบจำลองให้เหมาะสมกับลักษณะเฉพาะของพื้นที่ เช่น การพยากรณ์อุณหภูมิในเขตร้อนจะต่างจากเขตหนาวอย่างชัดเจน การรู้จักสถานที่ตั้งจึงเป็นสิ่งสำคัญในการทำให้แบบจำลองพยากรณ์มีความแม่นยำมากขึ้น</div><br>', unsafe_allow_html=True)
   
  
    
    st.markdown('<div class="highlight">อุณหภูมิ (Temperature)</div>', unsafe_allow_html=True)
    st.image("https://www.ucsf.edu/sites/default/files/2024-01/temperatures-card.jpg")
    st.markdown('<div class="text_indent">อุณหภูมิเป็นข้อมูลที่สำคัญที่สุดในการพยากรณ์ เพราะเป้าหมายหลักของระบบนี้คือต้องคาดการณ์อุณหภูมิในอนาคต ข้อมูลอุณหภูมิในอดีตจะช่วยให้เราสามารถสร้างแบบจำลองที่สามารถเรียนรู้จากเทรนด์และรูปแบบการเปลี่ยนแปลงของอุณหภูมิในแต่ละช่วงเวลาได้ โดยการทำการวิเคราะห์อุณหภูมิจะช่วยให้เราคาดการณ์การเปลี่ยนแปลงในอนาคตได้อย่างมีประสิทธิภาพ </div><br>', unsafe_allow_html=True)
   
    st.markdown('<div class="highlight">ความชื้น (Humidity)</div>', unsafe_allow_html=True)
    st.image("https://saveonenergy.ca/-/media/Images/SaveOnEnergy/residential/humidity-fix.jpg?la=en&h=667&w=1000&hash=76B37415B0115AEB2A7A15614F2F8423")
    st.markdown('<div class="text_indent"> ความชื้นในอากาศมีผลโดยตรงต่อการเปลี่ยนแปลงอุณหภูมิและสภาพอากาศโดยรวม การวิเคราะห์ข้อมูลความชื้นจะช่วยให้เราสามารถทำนายลักษณะของสภาพอากาศได้ดียิ่งขึ้น เช่น หากความชื้นสูงอาจทำให้เกิดการเกิดฝนหรือปรากฏการณ์อากาศที่เย็นลง การรู้จักความสัมพันธ์ระหว่างความชื้นและอุณหภูมิเป็นสิ่งสำคัญที่ช่วยให้การพยากรณ์มีความแม่นยำมากขึ้น</div><br>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight">ความเร็วลม (Wind Speed)</div>', unsafe_allow_html=True)
    st.image("https://windy.app//storage/posts/November2021/anemometer-wind-speed-measurement-instrument1.jpg")
    st.markdown('<div class="text_indent"> ข้อมูลความเร็วลมมีบทบาทสำคัญในการพยากรณ์อุณหภูมิ เนื่องจากลมมีผลต่อการกระจายความร้อนในอากาศ เช่น ลมเย็นอาจลดอุณหภูมิในพื้นที่หนึ่งได้หรือช่วยให้ลมร้อนขึ้นในบางพื้นที่ ข้อมูลนี้สามารถนำไปใช้ในการคำนวณหาความรู้สึกของอุณหภูมิที่แท้จริง (Wind Chill) หรือแม้กระทั่งการทำนายปรากฏการณ์อากาศเช่นพายุ การนำข้อมูลนี้มาใช้ในการวิเคราะห์จะทำให้ระบบสามารถพยากรณ์ได้แม่นยำยิ่งขึ้น</div><br>', unsafe_allow_html=True)
   
   
   
    
    st.markdown('<div class="big-font">การจัดการข้อมูลและการเลือกเทคนิคการวิเคราะห์ </div>', unsafe_allow_html=True)
    st.image("https://jobschiangrai.com/storage/uploads/image/5-data-science-and-analytics-jobs-for-freshers.jpg")
    st.markdown('<div class="highlight"> แหล่งที่มาของข้อมูล (Data Preprocessing)</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การวิเคราะห์และการสร้างโมเดลที่เกี่ยวข้องกับข้อมูลตลาดหุ้นนั้นต้องอาศัยข้อมูลจำนวนมาก ซึ่งต้องการความถูกต้องและความครบถ้วนของข้อมูลอย่างสูง อย่างไรก็ตาม การเตรียมข้อมูล (Data Preparation) ถือเป็นขั้นตอนที่สำคัญไม่น้อยไปกว่าการพัฒนาโมเดล เพราะข้อมูลที่ไม่สมบูรณ์หรือมีข้อบกพร่องอาจก่อให้เกิดความท้าทายในการทำงาน และส่งผลกระทบต่อคุณภาพของโมเดลที่สร้างขึ้นได้ </div><br>', unsafe_allow_html=True)

    st.markdown('<div class="text_indent"> ในโปรเจกต์นี้ เราจะมาดูวิธีการสร้าง Dataset ที่ใช้ในการพยากรณ์อุณหภูมิ โดยใช้เทคโนโลยี AI เช่น ChatGPT ในการจำลองข้อมูลที่เกี่ยวข้องกับตลาดหุ้น หรือในกรณีนี้คือข้อมูลสภาพอากาศ ซึ่งอาจมีความไม่สมบูรณ์บ้างจากแหล่งข้อมูลจริง การใช้เทคโนโลยี AI เพื่อสร้างข้อมูลจำลองนี้สามารถช่วยเสริมสร้างความเข้าใจและทักษะในการจัดการกับข้อมูลที่มีข้อบกพร่องอย่างมีประสิทธิภาพ เช่น การเติมค่าข้อมูลที่หายไป การตรวจสอบความผิดปกติของข้อมูล และการทำให้ข้อมูลมีความสมบูรณ์ก่อนที่จะนำไปใช้ในการสร้างโมเดลพยากรณ์อุณหภูมิ.</div><br>', unsafe_allow_html=True)
  
    st.markdown('<div class="text_indent">การเตรียมข้อมูลที่ดีจะช่วยให้แบบจำลองมีความแม่นยำมากยิ่งขึ้น และสามารถทำนายอุณหภูมิได้อย่างเชื่อถือได้ ซึ่งจะนำไปสู่การพัฒนาเครื่องมือที่ช่วยในการตัดสินใจในหลากหลายด้าน เช่น เกษตรกรรม พลังงาน และการท่องเที่ยวได้อย่างมีประสิทธิภาพ. </div><br>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight"> การสร้าง Dataset โดยใช้ ChatGPT</div>', unsafe_allow_html=True)
    st.image("https://motiongraphicplus.com/wp-content/uploads/2023/09/chat-gpt-logo-09.png")
    st.markdown('<div class="text_indent">ในการสร้างข้อมูลจำลองในโปรเจกต์นี้ เราได้ใช้ ChatGPT ในการสร้างข้อมูลจำลองจำนวน 250,000 รายการเกี่ยวกับสภาพอากาศของ 7 ประเทศและ 7 เมืองหลวง โดยกำหนดเงื่อนไขให้ข้อมูลมีลักษณะที่ไม่สมบูรณ์ (Incomplete Data) เพื่อให้สามารถนำไปใช้ในการฝึกและปรับปรุงขั้นตอนการเตรียมข้อมูลได้ ข้อมูลที่สร้างขึ้นประกอบด้วยคอลัมน์สำคัญดังนี้ </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> Date: วันที่ที่เกี่ยวข้องกับข้อมูลสภาพอากาศ ข้อมูลนี้บางส่วนอาจมีการขาดหาย (Missing Data) เพื่อจำลองสถานการณ์ที่ข้อมูลบางช่วงเวลาไม่สามารถบันทึกได้</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> Country: ประเทศที่เกี่ยวข้องกับข้อมูลสภาพอากาศ ข้อมูลนี้อาจมีข้อผิดพลาดในการกรอกข้อมูลหรือข้อมูลขาดหายบางส่วน โดยข้อมูลนี้ครอบคลุมถึง 7 ประเทศ ได้แก่ Thailand, UK, USA, South Korea, Japan, France, และ China</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> City: เมืองหลวงของแต่ละประเทศที่เกี่ยวข้องกับข้อมูลสภาพอากาศ ข้อมูลนี้อาจมีข้อผิดพลาดหรือข้อมูลบางจุดอาจขาดหาย เช่น Bangkok (Thailand), London (UK), New York (USA), Seoul (South Korea), Tokyo (Japan), Paris (France), และ Beijing (China)</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Temperature (°C): อุณหภูมิของเมืองในแต่ละวัน ข้อมูลนี้บางจุดอาจมีค่าผิดปกติหรือขาดหาย เช่น อุณหภูมิที่ไม่สมเหตุสมผลในบางวัน </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> Humidity (%): ความชื้นในอากาศ ข้อมูลบางชุดอาจขาดหายหรือมีค่าผิดปกติ เช่น ความชื้นที่สูงหรือต่ำเกินไป</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> Wind Speed (km/h): ความเร็วลม ข้อมูลนี้อาจมีค่าที่ขาดหายหรือผิดปกติ เช่น ความเร็วลมที่สูงเกินจริงหรือข้อมูลบางช่วงเวลาขาดหาย</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Moving Averages: การคำนวณค่าเฉลี่ยเคลื่อนที่ เช่น ค่าเฉลี่ย 50 วัน และ 200 วัน ซึ่งมีความสำคัญในการวิเคราะห์แนวโน้มของอุณหภูมิ ข้อมูลบางชุดอาจมีค่าไม่สมเหตุสมผลเนื่องจากความไม่สมบูรณ์ของข้อมูลดิบ </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> Change: การเปลี่ยนแปลงของอุณหภูมิจากวันก่อนหน้า ข้อมูลบางส่วนอาจมีค่าผิดปกติเนื่องจากข้อมูลวันที่ขาดหาย</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Percentage Change: การเปลี่ยนแปลงในรูปของเปอร์เซ็นต์จากอุณหภูมิวันก่อนหน้า ข้อมูลบางจุดอาจมีความไม่ถูกต้องหากข้อมูลอุณหภูมิไม่ครบถ้วน </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ข้อมูลจำลองเหล่านี้ช่วยให้เราได้ทดลองการจัดการข้อมูลที่มีข้อบกพร่องและพัฒนาเทคนิคการทำความสะอาดข้อมูล (Data Cleaning) รวมทั้งการเติมข้อมูลที่หายไป (Data Imputation) เพื่อให้สามารถนำไปใช้ในการพัฒนาโมเดลพยากรณ์อุณหภูมิได้อย่างมีประสิทธิภาพ </div><br>', unsafe_allow_html=True)
   
    
    st.markdown('<div class="highlight"> ประโยชน์ของการสร้างข้อมูลที่ไม่สมบูรณ์</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การสร้างข้อมูลที่ไม่สมบูรณ์ (Incomplete Data) มีประโยชน์หลายประการ โดยเฉพาะในแง่ของการพัฒนาและทดสอบระบบการจัดการข้อมูลที่มีข้อบกพร่อง การใช้ข้อมูลที่ไม่สมบูรณ์สามารถช่วยในการฝึกฝนและปรับปรุงขั้นตอนต่างๆ ในการเตรียมข้อมูล ซึ่งมีประโยชน์ดังนี้ </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">1. การฝึกฝนการจัดการข้อมูลที่ไม่สมบูรณ์: ข้อมูลที่ไม่สมบูรณ์สามารถจำลองสถานการณ์จริงที่มักเกิดขึ้นในโลกของข้อมูล เช่น ข้อมูลที่หายไปหรือข้อมูลที่ไม่ถูกต้อง ซึ่งช่วยให้สามารถพัฒนาเทคนิคในการจัดการข้อมูลเหล่านี้ได้อย่างมีประสิทธิภาพ เช่น การเติมข้อมูลที่หายไป (Data Imputation) หรือการลบข้อมูลที่ไม่สมบูรณ์ออก (Data Deletion)</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">2. การทดสอบและพัฒนาการทำความสะอาดข้อมูล: การมีข้อมูลที่ไม่สมบูรณ์ช่วยให้สามารถทดสอบและพัฒนาวิธีการทำความสะอาดข้อมูล (Data Cleaning) เพื่อให้ข้อมูลพร้อมสำหรับการวิเคราะห์และการสร้างโมเดลได้ดีขึ้น เช่น การตรวจจับค่าผิดปกติ (Outliers) หรือการแก้ไขข้อผิดพลาดที่เกิดจากการป้อนข้อมูลที่ไม่ถูกต้อง</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">3. การพัฒนาโมเดลที่ทนทานต่อข้อมูลที่ขาดหาย: การฝึกโมเดลด้วยข้อมูลที่ไม่สมบูรณ์ช่วยให้โมเดลนั้นสามารถทำงานได้ดีแม้ในกรณีที่ข้อมูลบางส่วนขาดหายไป ซึ่งสามารถช่วยให้โมเดลทำนายได้แม่นยำและมีความยืดหยุ่นในการทำงานในสถานการณ์จริง</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">4. การทดสอบการวิเคราะห์ข้อมูลในสถานการณ์ที่ท้าทาย: การใช้ข้อมูลที่ไม่สมบูรณ์ช่วยให้ผู้พัฒนาหรือผู้วิเคราะห์สามารถทดสอบการวิเคราะห์ข้อมูลในสถานการณ์ที่มีความท้าทาย เช่น ข้อมูลที่ขาดหายไป การมีข้อมูลที่ไม่สมบูรณ์สามารถช่วยให้เข้าใจว่าระบบหรือโมเดลที่พัฒนาขึ้นจะทำงานอย่างไรเมื่อเผชิญกับข้อมูลที่ไม่สมบูรณ์</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">5. การประเมินความสามารถในการจัดการกับข้อมูลที่ผิดปกติ: ข้อมูลที่มีความไม่สมบูรณ์ยังช่วยในการประเมินว่าระบบหรือเครื่องมือที่ใช้ในการเตรียมข้อมูลสามารถรับมือกับข้อมูลที่มีข้อผิดพลาดได้ดีแค่ไหน ซึ่งจะช่วยปรับปรุงเครื่องมือหรือขั้นตอนการทำงานในอนาคต </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">6. การสร้างระบบที่สามารถปรับตัวได้ดี: การสร้างข้อมูลที่ไม่สมบูรณ์ช่วยให้สามารถพัฒนาเครื่องมือและระบบที่สามารถจัดการกับข้อมูลที่ขาดหายหรือผิดปกติได้ดีขึ้น โดยทำให้ระบบสามารถปรับตัวและทำงานได้ในสถานการณ์ที่มีข้อมูลไม่สมบูรณ์หรือไม่ครบถ้วน </div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">การใช้ข้อมูลที่ไม่สมบูรณ์เป็นส่วนหนึ่งของกระบวนการพัฒนาและทดสอบช่วยให้สามารถจัดการกับความท้าทายในโลกของข้อมูลจริง และทำให้ระบบหรือโมเดลมีความยืดหยุ่นและทนทานต่อข้อผิดพลาดในข้อมูล </div><br>', unsafe_allow_html=True)
   
    
    #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)
   # st.markdown('<div class="text_indent"> </div><br>', unsafe_allow_html=True)

     #st.markdown('<div class="big-font"> </div>', unsafe_allow_html=True)

    #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)
    #st.markdown('<div class="text_indent"> </div><br>', unsafe_allow_html=True)
    
    #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)
    #st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)
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

        st.markdown('<div class="text_indent">ในการเตรียมข้อมูลสำหรับการวิเคราะห์อุณหภูมิและข้อมูลสภาพอากาศ การจัดการกับค่าที่ขาดหายไปในตารางข้อมูลที่มีคอลัมน์สำคัญ เช่น Date, Country, City, Temperature (°C), Humidity (%), Wind Speed (km/h) เป็นต้น เป็นขั้นตอนที่สำคัญอย่างยิ่ง เนื่องจากการจัดการกับข้อมูลที่หายไปมีผลต่อความแม่นยำในการพยากรณ์อุณหภูมิและการวิเคราะห์ในภายหลัง ดังนั้นจึงมีวิธีการที่สามารถนำไปใช้ได้หลายวิธี ขึ้นอยู่กับลักษณะของข้อมูลและวัตถุประสงค์ในการวิเคราะห์ โดยวิธีที่นิยมใช้ได้แก่ </div><br>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">การเติมค่าที่หายไปด้วยค่าเฉลี่ย (Mean Imputation)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">การเติมค่าที่หายไปด้วยค่าเฉลี่ยของแต่ละคอลัมน์ เช่น ค่าเฉลี่ยของอุณหภูมิ (Temperature) หรือความชื้น (Humidity) เป็นวิธีที่ง่ายและนิยมใช้กันมาก เนื่องจากช่วยให้ข้อมูลสมบูรณ์โดยไม่ต้องตัดข้อมูลออกจากการพิจารณา วิธีนี้เหมาะสมในกรณีที่ข้อมูลที่ขาดหายไปมีจำนวนน้อยและคาดว่าค่าที่หายไปไม่ส่งผลกระทบต่อการวิเคราะห์มากนัก </div><br>', unsafe_allow_html=True)

        st.markdown('<div class="highlight"> การใช้ค่าที่ใกล้เคียง (Forward Fill หรือ Backward Fill)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ในบางกรณี ค่าที่ขาดหายไปอาจเกิดจากการเก็บข้อมูลที่ไม่สมบูรณ์ในบางช่วงเวลา วิธีการเติมค่าที่หายไปด้วยค่าที่มีอยู่แล้วในแถวก่อนหน้าหรือแถวหลัง (เช่น ใช้ค่าอุณหภูมิจากวันก่อนหน้า หรือจากข้อมูลของเมืองเดียวกันเติมค่าที่หายไป) ช่วยรักษาความต่อเนื่องของข้อมูล และเหมาะสมในกรณีที่ข้อมูลมีลักษณะเป็นลำดับเวลา เช่น ข้อมูลอุณหภูมิที่สามารถคาดการณ์ได้จากการติดตามช่วงเวลา </div><br>', unsafe_allow_html=True)

        st.markdown('<div class="highlight">การตัดข้อมูลที่ขาดหายไป (Dropping Missing Data)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ในบางกรณี หากข้อมูลที่ขาดหายไปมีจำนวนน้อยและไม่ส่งผลกระทบต่อการวิเคราะห์ การตัดแถวที่มีค่าหายไปออกจาก DataFrame อาจเป็นวิธีที่เหมาะสม การตัดข้อมูลที่หายไปจะทำให้ข้อมูลที่เหลือมีความสมบูรณ์และไม่ต้องทำการเติมค่าผิดพลาดที่อาจทำให้การพยากรณ์ไม่แม่นยำ </div><br>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight"> การใช้ค่าที่คำนวณจากข้อมูลที่มีอยู่ (Model-Based Imputation)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">อีกหนึ่งวิธีที่สามารถใช้ในการเติมค่าที่หายไปคือการใช้โมเดลทางสถิติหรือโมเดลการเรียนรู้ของเครื่อง (Machine Learning) เพื่อคำนวณค่าที่หายไปจากข้อมูลที่มีอยู่ วิธีนี้มักใช้ในกรณีที่การขาดหายของข้อมูลเป็นลักษณะที่ไม่เป็นระเบียบ และสามารถสร้างแบบจำลองในการคาดเดาค่าที่หายไปได้จากข้อมูลที่มีอยู่ เช่น ใช้ข้อมูลของเมืองหรือประเทศเดียวกันในการคำนวณค่าสำหรับจุดที่ขาดหาย </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent"> การเลือกใช้วิธีใดในการจัดการค่าที่ขาดหายไปนั้น ขึ้นอยู่กับลักษณะของข้อมูลที่มีอยู่ ความสำคัญของข้อมูลที่หายไป และวัตถุประสงค์ในการพยากรณ์อุณหภูมิ การเลือกวิธีการที่เหมาะสมจะช่วยให้ข้อมูลมีความแม่นยำมากขึ้นและเหมาะสมสำหรับการนำไปสร้างแบบจำลองการพยากรณ์อุณหภูมิในขั้นตอนถัดไป</div><br>', unsafe_allow_html=True)
        

        st.markdown('<div class="big-font"> วิธีการเตรียมข้อมูล</div>', unsafe_allow_html=True)

      
        
        st.markdown('<div class="highlight">1.  ตรวจสอบคอลัมน์ที่ต้องการ</div>', unsafe_allow_html=True)
        code = '''
   required_columns = ['date', 'country', 'city', 'temperature (°C)', 'humidity (%)', 'wind_speed (km/h)']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• เราตรวจสอบว่าในข้อมูล (data) มีคอลัมน์ที่จำเป็นทั้งหมดหรือไม่ หากคอลัมน์ใดขาดหายไป จะมีการแจ้งข้อผิดพลาดให้ทราบ</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• คอลัมน์ที่จำเป็นสำหรับการพยากรณ์ในที่นี้ ได้แก่ date, country, city, temperature (°C), humidity (%), และ wind_speed (km/h)</div><br>', unsafe_allow_html=True)

        st.markdown('<br><div class="highlight">2. ลบคอลัมน์ที่ไม่จำเป็นสำหรับการพยากรณ์ </div>', unsafe_allow_html=True)
        code = '''
   X = data.drop(columns=['date'])
y_temp = data['temperature (°C)'].ffill()

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• เราลบคอลัมน์ date ออก เนื่องจากคอลัมน์นี้ไม่จำเป็นสำหรับการฝึกโมเดลหรือพยากรณ์ </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• คอลัมน์ temperature (°C) เป็นคอลัมน์เป้าหมายที่เราจะใช้ในการพยากรณ์ (ซึ่งจะถูกแยกออกไปเป็นตัวแปร y_temp) และเติมค่าที่หายไปในคอลัมน์นี้ด้วยวิธี forward fill (คือตัวล่าสุดที่มีจะถูกใช้แทนค่าที่ขาดหายไป) </div><br>', unsafe_allow_html=True)

        st.markdown('<br><div class="highlight">3. แปลงข้อมูล country และ city เป็นตัวเลข (One-Hot Encoding)</div>', unsafe_allow_html=True)
        code = '''
   data = pd.get_dummies(data, columns=['country', 'city'], drop_first=True)

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• การใช้ One-Hot Encoding เป็นวิธีการแปลงค่าหมวดหมู่ (เช่น country และ city) ให้เป็นตัวเลข โดยสร้างคอลัมน์ใหม่แทนค่าหมวดหมู่แต่ละประเภท</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การใช้ drop_first=True จะทำให้ไม่สร้างคอลัมน์ซ้ำ (จะหลีกเลี่ยงการสร้างคอลัมน์ที่ไม่จำเป็นเช่น city_New York และ city_Paris)</div><br>', unsafe_allow_html=True)


        st.markdown('<br><div class="highlight">4.  แปลงค่า True/False ให้เป็น 0/1</div>', unsafe_allow_html=True)
        code = '''
   bool_columns = data.select_dtypes(include=['bool']).columns
data[bool_columns] = data[bool_columns].astype(int)

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• ถ้าข้อมูลมีประเภท True/False (Boolean) อยู่ในตาราง เราจะแปลงค่าดังกล่าวเป็น 0 และ 1 เพื่อให้พร้อมใช้งานในโมเดล</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• วิธีนี้เหมาะสำหรับคอลัมน์ที่มีค่าบูลีน เช่น การเช็คว่าเหตุการณ์นั้นเกิดขึ้นหรือไม่</div><br>', unsafe_allow_html=True)


        st.markdown('<br><div class="highlight">5. แยกข้อมูลเป็นตัวเลขและข้อมูลหมวดหมู่</div>', unsafe_allow_html=True)
        code = '''
   numeric_cols = X.select_dtypes(include=['number']).columns  # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
categorical_cols = X.select_dtypes(exclude=['number']).columns  # เลือกเฉพาะคอลัมน์ที่เป็นข้อความ

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• เราจะแยกข้อมูลออกเป็นสองประเภท: ตัวเลข (numeric) และ หมวดหมู่ (categorical)</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• โดยการเลือกคอลัมน์ที่เป็นตัวเลขจากข้อมูล X และแยกคอลัมน์ที่ไม่ใช่ตัวเลขออกไป</div><br>', unsafe_allow_html=True)


        st.markdown('<br><div class="highlight">6. เติมค่าที่หายไปในข้อมูล</div>', unsafe_allow_html=True)
        code = '''
   num_imputer = SimpleImputer(strategy='mean')
X_numeric_imputed = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• สำหรับข้อมูลตัวเลข (Numeric):</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• เราจะใช้ SimpleImputer เพื่อเติมค่าที่หายไปในคอลัมน์ตัวเลขโดยการใช้ค่าเฉลี่ย (mean) ของคอลัมน์นั้นๆ</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• สำหรับข้อมูลหมวดหมู่ (Categorical)</div><br>', unsafe_allow_html=True)


        
        code = '''
   cat_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• สำหรับคอลัมน์ที่เป็นข้อมูลหมวดหมู่ เราจะใช้วิธีเติมค่าที่หายไปด้วยค่าที่พบบ่อยที่สุด (most frequent)</div><br>', unsafe_allow_html=True)
       


        st.markdown('<br><div class="highlight">7. รวมข้อมูลที่เติมค่ากลับมา</div>', unsafe_allow_html=True)
        code = '''
   X_imputed = pd.concat([X_numeric_imputed, X_categorical_imputed], axis=1)

        '''
        st.code(code, language='python')

        st.markdown('<div class="text_indent">• เมื่อเราทำการเติมค่าที่หายไปในข้อมูลทั้งตัวเลขและหมวดหมู่เสร็จแล้ว เราจะทำการรวมข้อมูลเหล่านั้นกลับมาใน DataFrame เดียว</div><br>', unsafe_allow_html=True)
       

        st.markdown('<br><div class="highlight">8. ตรวจสอบค่าที่หายไปหลังการเติมค่า</div>', unsafe_allow_html=True)
        code = '''
   missing_values = X_imputed.isnull().sum()
print(missing_values)

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">•เราตรวจสอบอีกครั้งว่าในข้อมูลที่ถูกเติมค่าแล้วยังมีค่าที่หายไปหรือไม่ (ควรเป็น 0 หมายความว่าไม่มีค่าที่หายไปแล้ว) </div><br>', unsafe_allow_html=True)
    

        st.markdown('<br><div class="highlight">9.  การปรับมาตรฐาน (Scaling)</div>', unsafe_allow_html=True)
        code = '''
   scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed[numeric_cols]), columns=numeric_cols)

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• เราจะใช้ StandardScaler เพื่อปรับขนาดข้อมูลที่เป็นตัวเลขให้มีค่าเฉลี่ย (mean) เท่ากับ 0 และส่วนเบี่ยงเบนมาตรฐาน (standard deviation) เท่ากับ 1 เพื่อให้การพยากรณ์มีประสิทธิภาพมากขึ้น</div><br>', unsafe_allow_html=True)
     
        st.markdown('<br><div class="highlight">10. แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ</div>', unsafe_allow_html=True)
       
        code = '''
   X_train, X_test, y_train_temp, y_test_temp = train_test_split(
    X_scaled, y_temp, train_size=0.2, random_state=42
)

        '''
        st.code(code, language='python')
        st.markdown('<div class="text_indent">• เราจะทำการแบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test) โดยใช้ train_test_split โดยแบ่งข้อมูล 80% สำหรับการฝึก (training) และ 20% สำหรับการทดสอบ (testing)</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การใช้ random_state=42 จะทำให้การแบ่งข้อมูลเป็นแบบสุ่มแต่สามารถทำซ้ำได้</div><br>', unsafe_allow_html=True)

      



        st.markdown('<div class="big-font"> โมเดลที่นำมาใช้</div>', unsafe_allow_html=True)

        st.markdown('<div class="highlight">RandomForestRegressor</div>', unsafe_allow_html=True)
        st.image("https://media5.datahacker.rs/2022/08/26-1024x761.jpg")
        st.markdown('<div class="text_indent">RandomForestRegressor เป็นเครื่องมือที่มีความยืดหยุ่นสูงและสามารถใช้ได้ดีในกรณีที่ข้อมูลมีความหลากหลาย เช่น ข้อมูลเกี่ยวกับอุณหภูมิที่ได้รับจากหลายแหล่งข้อมูล (เช่น อุณหภูมิในอดีต, ความชื้น, ความเร็วลม, ฯลฯ) </div><br>', unsafe_allow_html=True)
        
        st.markdown('<div class="normal-text">การนำไปใช้ในโปรเจค  </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• เตรียมข้อมูล: ข้อมูลเกี่ยวกับอุณหภูมิในอดีตและข้อมูลสภาพอากาศในอดีต (เช่น ความชื้น, ความเร็วลม) สามารถนำมาผสมผสานกันเป็นฟีเจอร์ในการฝึกโมเดล</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การฝึกโมเดล: ใช้ข้อมูลย้อนหลังในช่วงเวลาต่างๆ เพื่อฝึกฝนโมเดล โดยการใช้ Random Forest จะช่วยให้สามารถทำการประมวลผลข้อมูลที่มีความสัมพันธ์หลายมิติได้ดี</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การทำนาย: เมื่อฝึกโมเดลแล้ว สามารถนำข้อมูลสภาพอากาศที่กำลังเกิดขึ้นในขณะนั้นมาทำนายอุณหภูมิในอนาคต (เช่น อุณหภูมิในวันถัดไป)</div><br>', unsafe_allow_html=True)

        st.markdown('<div class="highlight">XGBRegressor ในการทำนายอุณหภูมิ</div>', unsafe_allow_html=True)
        st.image("https://i.sstatic.net/FHXO6.png")
        st.markdown('<div class="text_indent">XGBRegressor (Extreme Gradient Boosting) เป็นอัลกอริธึมที่มีประสิทธิภาพสูงในการทำนายข้อมูลที่ซับซ้อน โดยสามารถจัดการกับข้อมูลที่ไม่เป็นเชิงเส้นได้ดี ซึ่งเหมาะสมกับการทำนายอุณหภูมิในสภาพแวดล้อมที่มีความผันผวนสูง เช่น การทำนายอุณหภูมิในช่วงเวลาต่างๆ ที่อาจจะได้รับผลกระทบจากหลายปัจจัย </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="normal-text">การนำไปใช้ในโปรเจค  </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• เตรียมข้อมูล: เช่นเดียวกับ Random Forest, ใช้ข้อมูลของอุณหภูมิในอดีตและข้อมูลปัจจัยต่างๆ ที่มีผลต่ออุณหภูมิ (เช่น ความชื้น, ความเร็วลม) เพื่อฝึกโมเดล</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การฝึกโมเดล: XGBoost จะทำงานโดยการสร้างต้นไม้หลายๆ ต้นและเพิ่มประสิทธิภาพในการทำนายโดยใช้เทคนิค Gradient Boosting</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การทำนาย: สามารถใช้ XGBRegressor ในการทำนายอุณหภูมิในอนาคตโดยใช้ข้อมูลปัจจุบัน (เช่น ความชื้น, ความเร็วลม) และข้อมูลในอดีต</div><br>', unsafe_allow_html=True)

        st.markdown('<div class="highlight">Support Vector Regression (SVR) </div>', unsafe_allow_html=True)
        st.image("https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/05/support-vector-regression-svr.jpg?fit=1200%2C675&ssl=1")
        st.markdown('<div class="text_indent"> SVR (Support Vector Regression) ใช้แนวคิดจาก Support Vector Machine (SVM) เพื่อหาฟังก์ชันที่สามารถทำนายค่าอุณหภูมิได้ดีที่สุด โดยการหาค่าของ Hyperplane ที่ทำให้ค่าผลลัพธ์ของการทำนายมีความแม่นยำมากที่สุด</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="normal-text">การนำไปใช้ในโปรเจค  </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• เตรียมข้อมูล: ข้อมูลเช่น ความชื้น, ความเร็วลม, ความกดอากาศ, และอุณหภูมิในอดีตจะถูกใช้เป็นฟีเจอร์ในการฝึกโมเดล</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การฝึกโมเดล: SVR สามารถทำงานได้ดีในกรณีที่ข้อมูลมีลักษณะเป็น non-linear และสามารถทำนายอุณหภูมิที่มีความสัมพันธ์ซับซ้อนได้ดี</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• การทำนาย: เมื่อฝึกโมเดลแล้ว สามารถทำนายอุณหภูมิในอนาคตโดยใช้ข้อมูลต่างๆ ที่ได้จากการเก็บข้อมูลในปัจจุบัน</div><br>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">Ensemble Method</div>', unsafe_allow_html=True)
        st.image("https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/fc/4f/ensemble-learning-boosting.png")
        st.markdown('<div class="text_indent"> เป็นเทคนิคที่ใช้การรวมหลายโมเดลเพื่อเพิ่มความแม่นยำและประสิทธิภาพในการทำนายผล โดยการนำทั้งสามโมเดลที่ได้กล่าวถึงก่อนหน้า (RandomForestRegressor, XGBRegressor, และ Support Vector Regression (SVR)) มารวมกันจะช่วยเพิ่มความสามารถในการทำนายอุณหภูมิในโปรเจคของคุณ โดยการใช้ Ensemble Methods อย่าง Voting Regressor หรือ Stacking Regressor สามารถช่วยให้โมเดลมีประสิทธิภาพมากยิ่งขึ้นได้ </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">ทฤษฎีเบื้องหลัง Ensemble Method (Stacking)Ensemble Learning อาศัยหลักการที่ว่า การรวมหลาย ๆ โมเดลที่มีความแตกต่างกันสามารถช่วยให้ผลลัพธ์ที่ได้มีความแม่นยำมากขึ้น โดยการใช้หลายโมเดลเพื่อทำนายผลลัพธ์จะทำให้สามารถแก้ไขข้อผิดพลาดจากโมเดลใดโมเดลหนึ่งได้ เนื่องจากโมเดลแต่ละตัวอาจจะมีจุดแข็งและจุดอ่อนที่ต่างกัน Stacking เป็นหนึ่งในเทคนิค ensemble ที่ได้รับความนิยม โดยที่ base models หลายตัวจะทำการทำนายผลลัพธ์ และ meta-model จะนำผลลัพธ์จากโมเดลเหล่านั้นมาเรียนรู้เพื่อทำการทำนายที่แม่นยำยิ่งขึ้น การเรียนรู้แบบนี้ช่วยให้ระบบลดความเสี่ยงจากการใช้โมเดลเดียวที่อาจจะมีข้อผิดพลาด หรืออาจจะไม่สามารถจับลักษณะบางอย่างในข้อมูลได้</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="normal-text">การนำไปใช้ในโปรเจค  </div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• โมเดลทั้งสาม RandomForestRegressor, XGBRegressor, และ SVR จะถูกนำมารวมใน Voting Regressor โดยค่าทำนายจากแต่ละโมเดลจะถูกเฉลี่ยเพื่อนำมาทำนายอุณหภูมิในอนาคต</div><br>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">• เทคนิคนี้ช่วยลดความผิดพลาดจากโมเดลตัวเดียวและปรับให้ผลลัพธ์มีความเสถียรมากขึ้น</div><br>', unsafe_allow_html=True)


        datat = {
    'Model': ['Support Vector Regression (SVR)', 'Random Forest Regressor', 'XGBRegressor', 'Ensemble Method'],
    'ข้อดี': [
        'ใช้งานง่าย\nเหมาะสำหรับข้อมูลจำนวนน้อย\nทำงานได้ดีเมื่อมีรูปแบบที่ชัดเจน',
        'เข้าใจง่าย\nสามารถใช้กับทั้ง Regression และ Classification\nตีความได้ง่าย',
        'จัดการกับข้อมูลที่ไม่เป็นเส้นตรงได้ดี\nทำนายตัวเลขได้แม่นยำ',
        'รวมหลายโมเดลเพื่อเพิ่มความแม่นยำ\nลดความผิดพลาดจากโมเดลเดี่ยว'
    ],
    'ข้อเสีย': [
        'ใช้พลังในการคำนวณสูงเมื่อข้อมูลมาก\nไม่เหมาะกับข้อมูลที่มีมิติสูง',
        'อาจเกิด overfitting ถ้าโมเดลซับซ้อนเกินไป\nไม่เหมาะกับข้อมูลที่มีลักษณะซับซ้อนมาก',
        'การปรับพารามิเตอร์ซับซ้อน\nต้องการการคำนวณที่สูง',
        'ใช้เวลาในการฝึกและคำนวณมากขึ้น\nอาจยากต่อการตีความผลลัพธ์'
    ]
}

# สร้าง DataFrame จากข้อมูล
        dft = pd.DataFrame(datat)
        dft.index += 1

# แสดงตารางใน Streamlit
        st.markdown('<div class="highlight">ข้อดีและข้อเสียของแต่ละโมเดล</div>', unsafe_allow_html=True)
        st.table(dft)




        st.markdown('<div class="big-font"> การวัดประสิทธิภาพของโมเดล  </div>', unsafe_allow_html=True)

       
        st.markdown('<div class="text_indent">การวัดประสิทธิภาพของโมเดล Machine Learning เป็นขั้นตอนที่สำคัญในการประเมินความแม่นยำและความน่าเชื่อถือของการทำนายผล เพื่อให้เราแน่ใจว่าโมเดลที่สร้างขึ้นนั้นสามารถทำงานได้ตามที่คาดหวัง ในโปรเจกต์นี้ เราได้นำ Metrics หลายตัวมาใช้ในการประเมินผลลัพธ์จากโมเดลแต่ละตัวที่เราทดสอบ เพื่อช่วยให้การเลือกโมเดลที่ดีที่สุดในการทำนายอุณหภูมิเป็นไปอย่างมีประสิทธิภาพ โดย Metrics ที่เรานำมาใช้มีดังนี้ </div><br>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">RMSE (Root Mean Squared Error)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent"> RMSE เป็นการคำนวณค่าเฉลี่ยของความผิดพลาดยกกำลังสองและหาค่ารากที่สอง เพื่อให้มีหน่วยเหมือนกับข้อมูลที่แท้จริง ซึ่งจะช่วยให้การตีความความผิดพลาดง่ายขึ้นและสามารถเปรียบเทียบกับหน่วยข้อมูลได้</div><br>', unsafe_allow_html=True)
        st.latex(r'''RMSE = \sqrt{\frac{1}{n} \times \sum  (prediction - actual)^2}''')

        st.markdown('<div class="highlight">MAE (Mean Absolute Error)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">MAE ใช้ในการวัดความผิดพลาดโดยการหาค่าความผิดพลาดเฉลี่ยระหว่างค่าที่ทำนายและค่าจริง โดยไม่พิจารณาทิศทางของความผิดพลาด การใช้ MAE ช่วยให้เข้าใจความผิดพลาดที่เกิดขึ้นในแต่ละตัวอย่างได้ง่ายๆ </div><br>', unsafe_allow_html=True)
        st.latex(r'''MAE = \frac{1}{n} \sum \left| prediction_i - actual_i \right|''')

        st.markdown('<div class="highlight">R² (R-Squared)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">R² ใช้ในการวัดการกระจายของข้อมูลที่โมเดลสามารถอธิบายได้ โดยค่า R² จะอยู่ในช่วง 0 ถึง 1 ซึ่งถ้าใกล้ 1 แสดงว่าโมเดลสามารถอธิบายข้อมูลได้ดี หากใกล้ 0 แสดงว่าโมเดลไม่สามารถอธิบายข้อมูลได้เลย </div><br>', unsafe_allow_html=True)
        st.latex(r'''R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}''')

        st.markdown('<div class="highlight">MAPE (Mean Absolute Percentage Error)</div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent"> MAPE (Mean Absolute Percentage Error) เป็นอีกหนึ่ง Metric ที่ใช้ในการวัดความแม่นยำของโมเดลในการทำนาย โดยเฉพาะอย่างยิ่งในกรณีที่เราต้องการวัดความผิดพลาดในรูปแบบของเปอร์เซ็นต์ ซึ่งทำให้สามารถตีความผลลัพธ์ได้ง่ายและมีความหมายในแง่ของเปอร์เซ็นต์ความผิดพลาดที่เกิดขึ้นจากการทำนายค่าของโมเดลเมื่อเปรียบเทียบกับค่าจริง</div><br>', unsafe_allow_html=True)
        st.latex(r'''MAPE = \frac{1}{n} \sum \left| \frac{Actual - Forecast}{Actual} \right| \times 100''')

        st.markdown('<div class="highlight">5. Accuracy </div>', unsafe_allow_html=True)
        st.markdown('<div class="text_indent">Accuracy เป็นตัวชี้วัดที่คำนวณจากค่า MAPE เพื่อแปลงเป็นค่าความแม่นยำที่เข้าใจง่ายAccuracy=100−MAPE </div><br>', unsafe_allow_html=True)
        st.latex(r'''Accuracy = 100 - MAPE''')

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
