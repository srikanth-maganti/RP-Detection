import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
from PIL import Image

st.title("Detection of Retinitis Pigmentosa")
st.write("This is a web app to predict whether a person has Retinitis Pigmentosa or not.")
st.subheader("Please upload an image of the retina.")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

if(st.button("Predict")):
    if uploaded_file is not None:
        image=image.resize((224,224))
        image = np.array(image)
        image=np.expand_dims(image,axis=0)
        print(image.shape)
        prediction = predict(image)
        # if prediction == 0:
        #     
        # else:
        #     st.write("Prediction: Retinitis Pigmentosa")
        print(np.argmax(prediction))
        if np.argmax(prediction) == 0:
            st.markdown("## Your Normal :smiley:")
        else:
            st.write("## You have Retinitis Pigmentosa :disappointed: ")
        
       
    else:
        st.write("Please upload a valid image to predict.")