import streamlit as st
import torch
from PIL import Image
from model_loader import load_model_and_processor  


model_path = r'C:\Users\Lenovo\Downloads\image_captioning_model.pth'
processor, model = load_model_and_processor(model_path)


st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image to generate caption ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
   
    st.write("")
    st.write("Generating caption...")
    
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
        
    caption = processor.decode(output[0], skip_special_tokens=True)
    
   
    st.write("Generated Caption:", caption)
