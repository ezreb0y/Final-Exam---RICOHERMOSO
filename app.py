import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10
(_, _), (x_test, _) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cifar10_model.h5')
    return model

def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to match CIFAR-10 input size
    image = np.array(image)  # Convert PIL Image to NumPy array
    image = image / 255.0  # Normalize pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def import_and_predict(image_data, model):
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    return class_label

model = load_model()

st.write("# CIFAR-10 Image Classification/Prediction")
file = st.file_uploader("Select an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    class_label = import_and_predict(image, model)
    st.success("Predicted Class: " + class_label)
