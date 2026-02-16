import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- UPDATED LINE ---
# Now we load the .keras file you just created
model = tf.keras.models.load_model("flower_model.keras")
# --------------------

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.title("🌸 Flower Detection App")

file = st.file_uploader("Upload flower image")

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=400)

    # Preprocess
    img = img.resize((224, 224))
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)

    # Predict
    pred = model.predict(arr)
    
    # Display Results
    top_indices = np.argsort(pred[0])[::-1][:3]
    st.write("### Predictions:")
    for i in top_indices:
        confidence = pred[0][i] * 100
        st.write(f"**{class_names[i]}**: {confidence:.2f}%")
        # Simple progress bar for visual appeal
        st.progress(int(confidence))