import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


model.save('cifar10_model.h5')


model = tf.keras.models.load_model('cifar10_model.h5')


def preprocess_image(image):
    image = image.resize((32, 32))  # Resize the image to (32, 32)
    image = tf.keras.preprocessing.image.img_to_array(image)  # Convert PIL Image to numpy array
    processed_image = image / 255.0  # Normalize the image
    return processed_image


def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(tf.expand_dims(processed_image, axis=0))
    predicted_class = tf.argmax(prediction, axis=1)[0]
    return predicted_class


def main():
    st.title('CIFAR-10 Image Classification')
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        prediction = predict(image)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
