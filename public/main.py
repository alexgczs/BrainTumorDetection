from ast import arg
import imp
import streamlit as st
import time

if __name__ == '__main__':
    st.title("Brain Tumor Detection")
    image = st.file_uploader("Upload your photos", accept_multiple_files= False)
    if image != None:
        with st.spinner("processing"):
            time.sleep(3)
        st.image(image)