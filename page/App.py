import streamlit as st
from home import home
from machine_learning import machine_learning
from demo import demo

from NN import Neural_Network_Model
import os
from pathlib import Path
def local_css(file_name):
    file_path = Path(__file__).parent / file_name  
    if file_path.exists():  
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found: {file_path}")


local_css("styles.css")  

page = st.sidebar.radio("เลือกหน้า", ("Home", "machine learning", "Neural Network Model", "Demo Prediction"))


if page == "Home":
   home()


elif page == "machine learning":
    machine_learning()

elif page == "Neural Network Model":
    Neural_Network_Model()




elif page == "Demo Prediction":
    demo()
