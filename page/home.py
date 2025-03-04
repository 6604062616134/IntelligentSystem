import streamlit as st





def home():
    st.title("ทำไม AI ถึงสำคัญ?")
    st.image("https://cdn.sanity.io/images/ztw74qc4/production/515565a8ea2855fff9c7ad61244468cd83ab0180-1200x675.png?fit=max&auto=format", caption="รูปภาพจาก https://transcend.io/blog/ai-and-privacy")   
    st.markdown('<div class="text_indent">AI หรือ Artificial Intelligence (ปัญญาประดิษฐ์) คือเทคโนโลยีที่ทำให้ เครื่องจักรหรือระบบคอมพิวเตอร์สามารถเลียนแบบความคิด การตัดสินใจ และการเรียนรู้ได้เหมือนมนุษย์ มันเปรียบเสมือนสมองดิจิทัลที่ถูกออกแบบมาเพื่อช่วยให้ชีวิตของเราสะดวกสบายและฉลาดขึ้น</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ในปัจจุปัน  AI (Artificial Intelligence) หรือปัญญาประดิษฐ์ ได้กลายเป็นหัวข้อที่ได้รับความสนใจอย่างแพร่หลาย AI คือระบบหรือโปรแกรมที่ถูกออกแบบให้สามารถคิด วิเคราะห์ และตัดสินใจได้เหมือนมนุษย์ ไม่ว่าจะเป็นการแนะนำสินค้าในแพลตฟอร์มออนไลน์ ระบบผู้ช่วยอัจฉริยะอย่าง Siri หรือ Google Assistant ไปจนถึงการขับขี่รถยนต์อัตโนมัติ ทุกอย่างล้วนเกี่ยวข้องกับ AI ทั้งสิ้น </div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">แต่ AI ไม่ได้เป็นเพียงแค่ระบบที่คิดเองได้อย่างไร้เหตุผล ทุกอย่างถูกขับเคลื่อนด้วยศาสตร์ที่เรียกว่า "Machine Learning" ซึ่งทำให้ AI สามารถเรียนรู้และพัฒนาได้ด้วยตัวเอง</div>', unsafe_allow_html=True)
    
    st.markdown('<br><div class="big-font">Machine Learning</div>', unsafe_allow_html=True)
    st.image("https://miro.medium.com/v2/format:webp/0*_cgWPP25djXBauNZ.png", caption="รูปภาพจาก https://medium.com/investic/machine-learning-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3-fa8bf6663c07")   
    st.markdown('<div class="text_indent">Machine Learning (ML) เป็นแขนงหนึ่งของ AI ที่ช่วยให้คอมพิวเตอร์สามารถเรียนรู้จากข้อมูลและตัดสินใจได้โดยไม่ต้องมีการเขียนโปรแกรมแบบตายตัว หลักการของ Machine Learning คือการนำข้อมูลจำนวนมากมาวิเคราะห์เพื่อให้ได้รูปแบบ (pattern) ที่สามารถนำไปทำนายผลลัพธ์ในอนาคตได้</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">Machine Learning สามารถแบ่งออกเป็น 3 ประเภทหลัก ๆ ได้แก่</div>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">1. Supervised Learning - มีข้อมูลที่ถูกป้อนเข้ามาพร้อมกับผลลัพธ์ที่ถูกต้อง ตัวอย่างเช่น การจำแนกอีเมลเป็นสแปมหรือไม่สแปม</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">2. Unsupervised Learning - ไม่มีผลลัพธ์ที่แน่นอนให้คอมพิวเตอร์เรียนรู้ มักใช้ในการค้นหารูปแบบในข้อมูล เช่น การแบ่งกลุ่มลูกค้าตามพฤติกรรม</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">3. Reinforcement Learning - เป็นการเรียนรู้ผ่านการให้รางวัลหรือบทลงโทษ เช่น การพัฒนา AI ที่สามารถเล่นเกมและพัฒนาทักษะให้เก่งขึ้นเรื่อย ๆ</div><br>', unsafe_allow_html=True)
    

    st.markdown('<br><div class="big-font">การประเมินผลโมเดล Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"> ผลลัพธ์ของโมเดลจะถูกวิเคราะห์และประเมินด้วยตัวชี้วัด เพื่อค้นหาโมเดลที่มีประสิทธิภาพที่ดี ที่สุด โดยเป้าหมายหลักคือการนำโมเดลมาแก้ไขปัญาในสถานการณ์ จริง พร้อมทั้งเปรียบเทียบประสิทธิภาพในการทำงานระหว่างแต่ละโมเดล เพื่อหาแนวทางการประยุกต์ใช้งานโมเดลให้เหมาะสมที่สุด</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">การวัดประสิทธิภาพของโมเดล Machine Learning สามารถทำได้โดยใช้ตัวชี้วัดต่าง ๆ เช่น</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">MAE (Mean Absolute Error) - ค่าความคลาดเคลื่อนเฉลี่ยระหว่างค่าทำนายและค่าจริง</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">MSE (Mean Squared Error) - ค่าความคลาดเคลื่อนเฉลี่ยที่ยกกำลังสองเพื่อให้ค่าผิดพลาดที่มากมีผลกระทบมากขึ้น</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">R² (R-Squared Score) - ค่าที่บ่งบอกว่าโมเดลสามารถอธิบายความแปรปรวนของข้อมูลได้ดีแค่ไหน</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">MAPE (Mean Absolute Percentage Error) - ค่าความคลาดเคลื่อนสัมพัทธ์ในรูปเปอร์เซ็นต์</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Prediction Accuracy - ความแม่นยำของโมเดลในการทำนายผลลัพธ์</div><br>', unsafe_allow_html=True)


    st.markdown('<br><div class="big-font">โมเดลที่ได้นำมาใช้ในโปรเจคนี้</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">Random ForestRandom Forest เป็นอัลกอริธึมที่ใช้หลักการของ "Decision Trees" หลาย ๆ ต้นรวมกัน โดยแต่ละต้นไม้การตัดสินใจจะถูกฝึกด้วยข้อมูลตัวอย่างที่สุ่มมา (Bootstrapping) ทำให้มีความสามารถในการเรียนรู้ที่หลากหลายและลดโอกาสเกิด Overfitting ได้ดี</div><br>', unsafe_allow_html=True)
    st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2024/10/5.webp", caption="รูปภาพจาก https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/")  
   
    st.markdown('<div class="text_indent">XGBoost (Extreme Gradient Boosting)XGBoost เป็นอัลกอริธึมที่ได้รับความนิยมมากในการแข่งขันด้าน Machine Learning เนื่องจากมีประสิทธิภาพสูงและใช้ทรัพยากรอย่างมีประสิทธิภาพ หลักการทำงานของ XGBoost คือการสร้างต้นไม้หลาย ๆ ต้นที่เรียนรู้จากข้อผิดพลาดของต้นก่อนหน้า และปรับปรุงน้ำหนักให้ดีขึ้นในแต่ละรอบของการเรียนรู้</div><br>', unsafe_allow_html=True)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*Sc1OVfgV_jYTIDltlmirIw.png", caption="รูปภาพจาก https://medium.com/@shilpakamishetty/xgboost-350530593cb5")  
    
    st.markdown('<div class="text_indent">SVM (Support Vector Machine)SVM เป็นอัลกอริธึมที่ใช้ในการจำแนกข้อมูล โดยการสร้างเส้นแบ่ง (hyperplane) ที่สามารถแยกข้อมูลออกจากกันได้อย่างชัดเจน โดยมีแนวคิดพื้นฐานคือการหาตำแหน่งของเส้นแบ่งที่ทำให้เกิด Margin (ระยะห่างระหว่างกลุ่มข้อมูล) มากที่สุด ซึ่งช่วยเพิ่มความสามารถในการจำแนกประเภทของข้อมูลใหม่ได้ดีขึ้น</div><br>', unsafe_allow_html=True)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*Q6cnY4xiPuESjpC99vNdLg.png", caption="รูปภาพจาก https://medium.com/@skillcate/support-vector-machine-svm-simplest-intuition-4f19028efbd0")  
    
    st.markdown('<div class="text_indent">Ensemble Learning: หลักการรวมโมเดลเพื่อให้ได้ผลลัพธ์ที่ดีที่สุด Ensemble Learning เป็นเทคนิคที่รวมเอาหลายโมเดลมาทำงานร่วมกันเพื่อเพิ่มประสิทธิภาพให้กับระบบ โดยมีแนวทางหลัก ๆ ดังนี้</div><br>', unsafe_allow_html=True)
    st.image("https://images.prismic.io/encord/4fda620b-ac6c-45dc-ba17-f0d68bc7888f_What+is+Ensemble+Learning_.png?auto=compress%2Cformat&fit=max", caption="รูปภาพจาก https://encord.com/blog/what-is-ensemble-learning/")  
    
    st.markdown('<div class="normal-text">-Bagging - ใช้หลายโมเดลที่เหมือนกัน แต่ฝึกด้วยข้อมูลที่แตกต่างกัน เช่น Random Forest</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">-Boosting - ใช้หลายโมเดลที่ต่อเนื่องกัน โดยโมเดลใหม่จะเรียนรู้จากความผิดพลาดของโมเดลก่อนหน้า เช่น XGBoost</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="normal-text">-Stacking - ใช้โมเดลหลายตัวแล้วนำผลลัพธ์มาผ่านโมเดลขั้นสุดท้ายเพื่อปรับปรุงผลลัพธ์ให้ดีขึ้น</div><br>', unsafe_allow_html=True)

   
    st.markdown('<div class="text_indent">Convolutional Neural Networks (CNN) กับการประมวลผลเสียง </div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">CNN ไม่ได้ถูกใช้แค่กับการรู้จำภาพเท่านั้น แต่ยังสามารถนำมาใช้วิเคราะห์และจำแนกเสียงได้ด้วย โดยทำงานร่วมกับการแปลงเสียงเป็น Spectrogram ซึ่งเป็นภาพที่แสดงพลังงานของคลื่นเสียงในแต่ละความถี่ตามเวลา โครงสร้างของ CNN ในการประมวลผลเสียงประกอบด้วย</div><br>', unsafe_allow_html=True)
    st.image("https://www.guru99.com/images/tensorflow/083018_0542_WhatisDeepl3.png", caption="รูปภาพจาก https://thaiprogrammer.org/deep-learning-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3/")  
    
    st.markdown('<div class="big-font">การนำ Machine Learing มาประยุกต์ใช้ในการทำโปรเจค</div>', unsafe_allow_html=True)
    st.markdown('<div class="highlight">การพยากรณ์อุณหภูมิโดยใช้ข้อมูลสภาพอากาศ</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ในโลกของการวิเคราะห์ข้อมูล การพยากรณ์อุณหภูมิที่แม่นยำถือเป็นกุญแจสำคัญในการวางแผนและตัดสินใจในหลายด้าน เช่น เกษตรกรรม พลังงาน และการท่องเที่ยว โปรเจกต์นี้มุ่งเน้นไปที่การพัฒนาระบบพยากรณ์อุณหภูมิ โดยใช้ข้อมูลสภาพอากาศที่สำคัญ ได้แก่ ประเทศ (country), เมือง (city), อุณหภูมิ (temperature °C), ความชื้น (humidity %), และความเร็วลม (wind speed km/h) ซึ่งข้อมูลเหล่านี้จะถูกนำมาวิเคราะห์เพื่อสร้างแบบจำลองที่สามารถคาดการณ์อุณหภูมิได้อย่างแม่นยำ</div><br>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight">กระบวนการและเทคนิคที่ใช้ในการพยากรณ์</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ข้อมูลสภาพอากาศที่รวบรวมจะถูก ประมวลผลและปรับแต่ง (Preprocessing) เพื่อให้เหมาะสมสำหรับการวิเคราะห์และพยากรณ์อุณหภูมิ ขั้นตอนนี้รวมถึงการจัดการค่าขาดหาย การปรับขนาดข้อมูล และการแปลงข้อมูลให้อยู่ในรูปแบบที่เหมาะสม จากนั้นจะนำ โมเดล Machine Learning ที่มีประสิทธิภาพสูง มาประยุกต์ใช้เพื่อพัฒนาแบบจำลองที่สามารถทำนายอุณหภูมิได้อย่างแม่นยำ</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ข้อมูลเหล่านี้จะถูก จัดเตรียมและปรับแต่ง (Preprocessing) เพื่อให้เหมาะสมกับการวิเคราะห์และการทำนาย จากนั้นจะนำ โมเดล Machine Learning ที่มีประสิทธิภาพสูง มาใช้ในการประมวลผลและทำนาย เช่น<br><br>'
                '🔍 Support Vector Machine (SVM): หาความคล้ายคลึงของข้อมูลเพื่อนบ้านใกล้เคียงเพื่อทำนายราคา<br><br>'
                '🌳 Random ForestRandom Forest: ตัดสินใจแบบลำดับขั้นตอนจากข้อมูลที่ป้อนเข้า<br><br>'
                '🌳 Extreme Gradient Boosting (XGBoost): ช่วยทำนายข้อมูลที่มีความต่อเนื่องด้วยเส้นที่เหมาะสมที่สุด<br><br>'
                '🧩 Ensemble Method (Stacking): รวมโมเดลหลายตัวเข้าด้วยกันเพื่อเพิ่มประสิทธิภาพการทำนาย <br><br></div>', unsafe_allow_html=True)
   
    st.markdown('<div class="big-font">การนำ CNN ประยุกต์ใช้ในการทำโปรเจค</div>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent">ในยุคที่เทคโนโลยีการประมวลผลเสียงและการเรียนรู้ของเครื่องมีการพัฒนาอย่างรวดเร็ว การนำ Convolutional Neural Networks (CNN) มาใช้ในการจำแนกเสียงสัตว์เลี้ยง เช่น เสียงของสุนัขและแมว กลายเป็นเครื่องมือที่มีความสำคัญในการแยกแยะและระบุเสียงได้อย่างแม่นยำ โดยปกติแล้ว CNN จะถูกใช้ในงานประมวลผลภาพ แต่ในกรณีนี้ได้มีการนำมาใช้ในการวิเคราะห์เสียงโดยการแปลงสัญญาณเสียงเป็น Spectrogram หรือ MFCCs (Mel-Frequency Cepstral Coefficients) ซึ่งมีลักษณะคล้ายภาพสองมิติ ช่วยให้ CNN สามารถประมวลผลและจำแนกเสียงจากรูปแบบที่แสดงผลได้ดียิ่งขึ้น</div><br>', unsafe_allow_html=True)
    st.markdown('<div class="highlight">กระบวนการและเทคนิคที่ใช้ในการจำแนกเสียง</div>', unsafe_allow_html=True)

    st.markdown('<div class="text_indent">ในการจำแนกเสียงของสุนัขและแมวด้วย Convolutional Neural Networks (CNN) จะมีการใช้กระบวนการและเทคนิคต่าง ๆ ที่สำคัญเพื่อให้การทำนายแม่นยำ ได้แก่<br><br>'
                '🔊 การแปลงสัญญาณเสียงเป็น Spectrogram หรือ MFCCsการแปลงสัญญาณเสียงเป็น Spectrogram หรือ MFCCs ช่วยในการเปลี่ยนเสียงเป็นรูปแบบที่ CNN สามารถประมวลผลได้<br><br>'
                '🧠 การออกแบบโครงสร้างของ CNNการออกแบบเครือข่าย CNN ที่เหมาะสมเพื่อจำแนกเสียงจากลักษณะของสัญญาณเสียงที่แตกต่างกันระหว่างสุนัขและแมว<br><br>'
                '📊 การฝึกและประเมินโมเดลการฝึกโมเดลด้วยข้อมูลเสียงที่มีฉลาก (เช่น เสียงของสุนัขและแมว) และการประเมินผลเพื่อให้แน่ใจว่าโมเดลสามารถแยกเสียงได้อย่างแม่นยำ<br><br>'
                '🔧 การปรับปรุงโมเดลการทดสอบและปรับแต่งโมเดลเพื่อเพิ่มประสิทธิภาพในการจำแนกเสียงให้ดียิ่งขึ้น <br><br></div>', unsafe_allow_html=True)

    st.markdown('<div class="text_indent"></div><br>', unsafe_allow_html=True)
    st.markdown('<div class="text_indent"></div><br>', unsafe_allow_html=True)
   
  