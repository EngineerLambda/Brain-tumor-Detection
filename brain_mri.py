import streamlit as st
import tensorflow as tf
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


# Loading the pre-saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(os.path.join(cwd,"resources","brain_mri.h5"))

model = load_model()

image_file = st.file_uploader("Upload Image below", type = ['jfif','jpg','png','jpeg'])
if image_file is not None:
 # Preprocess the image
    if not "image_file" in st.session_state:
        st.session_state["image_file"] = image_file

    def preprocess_image(file_path):
        try:
            img_dir = os.path.join(cwd,file_path)
            img = cv2.imread(img_dir)
            img_resize = cv2.resize(img, (224,224))
            return img_resize
        except:
            st.error("Image could not be read")


    # Creating function for the predict botton
    def predict():
        image_predict = preprocess_image(st.session_state["image_file"].name)
        pred = model.predict(image_predict.reshape(-1,224,224,3))
        class_idx = int(pred[0][0])
        if class_idx == 0:
            return st.success("This patient doesn\'t have brain tumor")
        else:
            return st.error("This patient has brain tumor")


    # Validating whether an image has been uploaded and showing the PREDICT
    # button if value of image is True and not None     
    if image_file:
        img_to_show = preprocess_image(st.session_state["image_file"].name)
        st.image(img_to_show)
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
