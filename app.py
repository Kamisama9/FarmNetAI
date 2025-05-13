import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

# Define custom GRL layer
class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -dy
            return x, grad
        return _reverse(x)

# Load full model and extract feature model
full_model = load_model(
    r"final_modelv1.h5",
    custom_objects={'GradientReversal': GradientReversal}
)

# Replace 'feature' with actual layer name from your model
feature_extractor = tf.keras.Model(
    inputs=full_model.input,
    outputs=full_model.get_layer("feature_vec").output  # Ensure this is correct
)

# Load KNN and class names
with open(r"Model_with_KNN.pkl", 'rb') as f:
    knn_data = pickle.load(f)

knn_classifier = knn_data['model']
class_names = knn_data['class_names']

# Title
st.title("🌾 Wheat Leaf Disease Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Layout: two columns
    col1, col2 = st.columns([1, 1])

    # Column 1: Show image
    with col1:
        img = Image.open(uploaded_file).resize((224, 224))
        st.image(img, caption="Uploaded Image", use_container_width =True)

    # Column 2: Prediction controls and result
    with col2:
        if st.button("Predict"):
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            features = feature_extractor.predict(img_array)
            pred_index = knn_classifier.predict(features)[0]
            confidence = knn_classifier.predict_proba(features)[0][pred_index]

            if confidence >= 0.80:
                raw_class = class_names[pred_index]
                clean_class = raw_class.replace('_', ' ').replace('test', '').strip().title()
                st.markdown(f"**Predicted Class:** {clean_class}")
                st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
            else:
                st.warning("Prediction confidence is below 80%. Unable to determine the disease confidently.")
