from operator import mod
import streamlit as st
import time
from PIL import Image
import numpy as np
from model import BrainTumorDetector
from googleViT import query
model = BrainTumorDetector()

if __name__ == "__main__":
    st.title("Brain Tumor Detection")
    image = st.file_uploader("Upload your photos", accept_multiple_files=False)
    if image != None:
        with st.spinner("processing"):
            time.sleep(0.1)
        image_np = np.array(Image.open(image))
        
        # response = query(Image.open(image).tobytes())

        response = model.evaluate(image_np)
        st.info(
            f"{response}"
        )

        st.button("Explain result")
