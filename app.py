import streamlit as st 
from PIL import Image
from main import mainAgent

st.title('Cureify: Clinical Decision Support System.')

prompt = st.text_input('Enter your prompt')
img = st.file_uploader('Upload an image')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
result = mainAgent(prompt, img)

if st.button('Submit'):
    st.write(result)
