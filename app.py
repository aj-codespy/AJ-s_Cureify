import streamlit as st
from main import mainAgent

st.title("Cureify - Clinical Decision Support System")

prompt = st.text_area("Enter your symptoms:")
img = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])

if st.button('Submit'):
    if not prompt and not img:
        st.warning("Please enter symptoms or upload an image before submitting.")
    else:
        result = mainAgent(prompt, img)
        st.write(result)
