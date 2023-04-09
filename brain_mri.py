import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import time 

# Getting the current working directory
cwd = os.getcwd()

# Creating title for the web app
st.title("BRAIN TUMOR DETECTION")
st.write("""
### *Using machine learning*
""")

# Creating side bar
with st.sidebar:
    st.title("Brain MRI status check")
    with Image.open(os.path.join(cwd,"resources","brainimg.png")) as pic:
        st.image(pic)
    st.info(
        """This web app is using a pre-trained machine learning model
            to determine whether a patient has brain tumor or not
            from the image of their brain MRI scan"""
            )


# Loading the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(os.path.join(cwd,"resources","brain_mri.h5"))

model = load_model()

image_file = st.file_uploader("Upload Image below", type = ['jfif','jpg','png','jpeg'])
if image_file is not None:
 # Preprocess the image
    def preprocess_image():
        try:
        # img_dir = os.path.join(cwd, image_file.name)
            # Read the uploaded file as an image
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            img_resize = cv2.resize(img, (224,224))
            return img_resize
        except:
            st.error("Image could not be read")


    img_to_show = preprocess_image()
    st.image(img_to_show)
    # Creating function for the predict botton
    def predict():
        image_predict = img_to_show
        pred = model.predict(image_predict.reshape(-1,224,224,3))
        class_idx = int(pred[0][0])
        if class_idx == 0:
            return st.success("This patient doesn\'t have brain tumor")
        else:
            return st.error("This patient has brain tumor")


    # Validating whether an image has been uploaded and showing the PREDICT
    # button if value of image is True and not None     
    
    with st.spinner("Getting the model to work, just for you ..."):
        time.sleep(1)
        if st.button('Detect'):
            predict()
else:
    st.error("Kindly upload a valid image")


# Creating a custom footer using css\
# I will render it as a markdown in streamlit
footer="""
<style>
a:link , a:visited{
color: blue;
background-color: transparent;
}

a:hover,  a:active {
color: red;
background-color: transparent;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with streamlit by:<a style='display: block; text-align: center;' href="https://www.github.com/EngineerLambda" target="_blank">Jimoh Abdulsomad Abiola(EngineerLambda)</a></p>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)
