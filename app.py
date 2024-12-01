import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Load the model from the h5 file
model = keras.models.load_model('model.h5')

# Streamlit app
st.title("Image Classification")
st.write("Upload an image for classification.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
class_names=['Normal','Tuberculosis']
# Perform classification when an image is uploaded
if uploaded_file is not None:
    # Read and preprocess the image
    image = tf.image.decode_image(uploaded_file.read(), channels=3)
    image = tf.image.resize(image, (224, 224))  # Assuming your model expects 224x224 input
    image = image / 255.0  # Normalize the pixel values (if required by your model)

    # Make predictions
    predictions = model.predict(tf.expand_dims(image, axis=0))
    class_index = tf.argmax(predictions, axis=1)[0]

    # Display the predicted class and confidence
    st.write("Predicted Class:")
    st.write("Confidence:")
    st.write(predictions[0][class_index])
    if predictions[0][class_index]<float(0.5):
        st.write(class_names[0])
    else:
        st.write(class_names[1])