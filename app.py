import streamlit as st 
from PIL import Image
from main import mainAgent

st.title('Cureify: Clinical Decision Support System.')

prompt = st.text_input('Enter your prompt')
img = st.file_uploader('Upload an image')
if img is not None:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
result = mainAgent(prompt, img)

if st.button('Submit'):
    st.write(result)
