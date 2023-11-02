import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import streamlit as st

import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 'truetype'})

from matplotlib import offsetbox
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob
import ntpath
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
import scipy as sc
from streamlit.components.v1 import components



image_paths = glob.glob('images/*.jpg')


images = {}
for image_path in image_paths:
    image = cv2.imread(image_path, 3)
    b,g,r = cv2.split(image)           # get b, g, r
    image = cv2.merge([r,g,b])         # switch it to r, g, b
    image = cv2.resize(image, (200, 200))
    images[ntpath.basename(image_path)] = image
    
    
def load_image(image):
    image = plt.imread(image)
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize(img, [400, 400])
    img = img[tf.newaxis, :] # shape -> (batch_size, h, w, d)
    return img

#
# content layers describe the image subject
#
content_layers = ['block5_conv2']

#
# style layers describe the image style
# we exclude the upper level layes to focus on small-size style details
#
style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        #'block4_conv1',
        #'block5_conv1'
    ]

def selected_layers_model(layer_names, baseline_model):
    outputs = [baseline_model.get_layer(name).output for name in layer_names]
    model = Model([vgg.input], outputs)
    return model

# style embedding is computed as concatenation of gram matrices of the style layers
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleModel, self).__init__()
        self.vgg =  selected_layers_model(style_layers + content_layers, vgg)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # scale back the pixel values
        inputs = inputs*255.0
        # preprocess them with respect to VGG19 stats
        preprocessed_input = preprocess_input(inputs)
        # pass through the reduced network
        outputs = self.vgg(preprocessed_input)
        # segregate the style and content representations
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # calculate the gram matrix for each layer
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # assign the content representation and gram matrix in
        # a layer by layer fashion in dicts
        content_dict = {content_name:value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}

vgg = model = tf.keras.models.load_model('vgg.h5')

def image_to_style(image_tensor):
    extractor = StyleModel(style_layers, content_layers)
    return extractor(image_tensor)['style']

def style_to_vec(style):
    # concatenate gram matrics in a flat vector
    return np.hstack([np.ravel(s) for s in style.values()])

#
# Print shapes of the style layers and embeddings
#
image_tensor = load_image(image_paths[0])
style_tensors = image_to_style(image_tensor)
for k,v in style_tensors.items():
    print(f'Style tensor {k}: {v.shape}')
style_embedding = style_to_vec( style_tensors )
print(f'Style embedding: {style_embedding.shape}')

#
# compute styles
#
image_style_embeddings = {}
for image_path in tqdm(image_paths):
    image_tensor = load_image(image_path)
    style = style_to_vec( image_to_style(image_tensor) )
    image_style_embeddings[ntpath.basename(image_path)] = style    
    
    
def search_by_style(image_style_embeddings, reference_image):
    v0 = image_style_embeddings[reference_image]
    distances = {}
    for k,v in image_style_embeddings.items():
        d = sc.spatial.distance.cosine(v0, v)
        distances[k] = d

    sorted_neighbors = sorted(distances.items(), key=lambda x: x[1], reverse=False)
    return sorted_neighbors
    
st.title('Visual Search App')

st.markdown(
    """
    <p style="font-size:20px;">This application is designed to help you explore and find images with similar artistic styles. 
    <strong>Using our application is simple!</strong>
    You can upload an image and instantly discover a curated selection of visually similar artwork.</p>
    """,
    unsafe_allow_html=True
)

st.write('Upload an image to perform a visual search.')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

x = st.number_input('Choose the number of similar images to be displayed',min_value=1,max_value=9,value=3)

button_clicked = st.button('Submit', key=1002)
if button_clicked:
 st.write('Discovering similar artistic styles...')
 if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Here's the image you've uploaded")
    st.image(image, use_column_width=True)
    image_path = os.path.join('images/',uploaded_file.name)
    image.save(image_path)
    
    
    sorted_neighbors = search_by_style(image_style_embeddings, uploaded_file.name)
    
    # Create a list of similar images
    similar_images = [Image.open('images/' + sorted_neighbors[i][0]) for i in range(1, x+1)]  
    if x==1: 
     st.write("Here is an image with similar artistic style")  
    else:    
     st.write(f'Here are {x} images with similar artistic style ')
    # Display similar images as a slideshow
    for img in similar_images:
        st.image(img, use_column_width=True)  
