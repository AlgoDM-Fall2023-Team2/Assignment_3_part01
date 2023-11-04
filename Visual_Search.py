import streamlit as st
import numpy as np
from functions.computations import *
from PIL import Image
import scipy as sc
import json

# path to JSON file
file_path = 'image_style_embeddings.json'

# Open and read the JSON file
with open(file_path, 'r') as f:
    loaded_image_style_embeddings = json.load(f)




st.title('Visual Search App')

st.markdown(
    """
    <p style="font-size:20px;">This application is designed to help you explore and find images with similar artistic styles. 
    <strong>Using our application is simple!</strong>
    You can upload any image and instantly discover desired number of visually similar artwork.</p>
    """,
    unsafe_allow_html=True
)

st.write('Upload an image to perform a visual search.')

image = st.file_uploader("Upload Image", type = ['jpg'])
n = st.number_input('Choose the number of similar images to be displayed',min_value=1,max_value=10,value=3)
button_clicked = st.button('Predict', key=1002)
   
if button_clicked and image is not None:

    st.header("The Image you uploaded")

    st.image(image=image)
    
    v0 = style_to_vec(image_to_style(load_image(image)))
    v0 = v0.tolist()
    distances = {}
    for k,v in loaded_image_style_embeddings.items():
        d = sc.spatial.distance.cosine(v0, v)
        distances[k] = d

    sorted_neighbors = sorted(distances.items(), key=lambda x: x[1], reverse=False)
    # Create a list of similar images
    similar_images = [Image.open('images/' + sorted_neighbors[i][0]) for i in range(1, n+1)]  
    if n==1: 
     st.header("Here is an image with similar artistic style")  
    else:    
     st.header(f'Here are {n} images with similar artistic style ')
    # Display similar images 
    for img in similar_images:
        st.image(img, use_column_width=True) 
        
    