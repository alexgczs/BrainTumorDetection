import streamlit as st
import time
from BrainTumorDetector import BrainTumorDetector

def start_detection(images):
    model = BrainTumorDetector()
    response = model.evaluate(images)

if __name__ == '__main__':
    st.title("Brain Tumor Detection")
    images = st.file_uploader("Upload your photos", accept_multiple_files= True)

    if images != None:
        start_detection(images)