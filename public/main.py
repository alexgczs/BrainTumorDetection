import streamlit as st
import time
from BrainTumorDetector import BrainTumorDetector
from PIL import Image

model = BrainTumorDetector()

def start_detection(images):
    if images == []: 
        return

    c1_decision, c2_decision, c3_decision = st.columns(3) 
    with c2_decision:
        with st.spinner("Calculating"):
            response = model.evaluate(images)
    st.info(
        f"{response}"
    )

def sample_flow(pacient_index):
    c1_image, c2_image, c3_image = st.columns(3)
    columns = [c1_image, c2_image, c3_image]
    images = []
    for index in range(3):
        image_path = f"data/paciente{pacient_index}/p{pacient_index}_{index}.jpeg"
        images.append(image_path)
        column = columns[index]
        column.image(Image.open(image_path))
    
    start_detection(images)

def headers():
    st.container()
    st.title("Brain Tumor Detection 🧠")
    images = st.file_uploader("Upload your photos", accept_multiple_files= True)
    if images != None:
        start_detection(images)
    _, c1_main,_, c2_main,_= st.columns(5)
    if c1_main.button("Data from pacient 1", 1):
        sample_flow(0)
    if c2_main.button("Data from pacient 2", 2):
        sample_flow(1)

if __name__ == '__main__':
    headers()
    