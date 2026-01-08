import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    .stSidebar { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .black-text { color: black; font-weight: bold; font-size: 24px; }
    .small-black-text { color: black; font-size: 18px; }
    .progress-bar-container { width: 100%; background-color: #eee; border-radius: 5px; height: 25px; }
    .progress-bar { height: 100%; border-radius: 5px; text-align: center; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model('best_cat_dog_model.h5', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

IMG_SIZE = 224
def prepare_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

st.sidebar.title("🖼️ Image Source")
mode = st.sidebar.radio("Input Selection:", ["Example Gallery", "Upload Image"])
input_image = None

if mode == "Example Gallery":
    example_folder = "examples"
    if os.path.exists(example_folder):
        files = [f for f in os.listdir(example_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if files:
            selected_name = st.sidebar.selectbox("Pick an example:", files)
            input_image = Image.open(os.path.join(example_folder, selected_name))
        else:
            st.sidebar.warning("No images found in /examples folder.")
    else:
        st.sidebar.error("Folder 'examples/' not found.")
else:
    uploaded_file = st.sidebar.file_uploader("Choose a cat or dog photo...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        input_image = Image.open(uploaded_file)

st.markdown('<p class="black-text">🐾 Cat vs Dog Prediction</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    if input_image:
        st.image(input_image, caption="Current Selection", use_column_width=True)
    else:
        st.info("Please select an image from the sidebar to begin.")
    predict_clicked = st.button("Predict")

with col2:
    if predict_clicked:
        if input_image is None:
             st.markdown('<p style="color:red; font-weight:bold;">⚠️ No image selected. Please upload or choose an image first.</p>', unsafe_allow_html=True)
        elif model:
            with st.spinner("Analyzing image features..."):
                processed_img = prepare_image(input_image)
                prediction = model.predict(processed_img)[0][0]

                prob_dog = float(prediction)
                prob_cat = 1 - prob_dog
                label = "Dog" if prob_dog > 0.5 else "Cat"
                final_score = prob_dog if label == "Dog" else prob_cat

                st.markdown(f'<p class="black-text">Result: {label}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="small-black-text">Confidence Level: {final_score*100:.2f}%</p>', unsafe_allow_html=True)

                bar_color = "#1f77b4" if label == "Cat" else "#ff7f0e"
                st.markdown(f"""
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width:{final_score*100}%; background-color:{bar_color}">
                            {final_score*100:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Model not detected. Ensure 'best_cat_dog_model.h5' is in the root directory.")

st.markdown("---")
with st.expander("Model Parameters & Preprocessing Info"):
    st.markdown(f"""
        <p class="small-black-text"><b>Target Size:</b> {IMG_SIZE}x{IMG_SIZE}</p>
        <p class="small-black-text"><b>Architecture:</b> EfficientNet (Transfer Learning)</p>
        <p class="small-black-text"><b>Preprocessing:</b> EfficientNet-specific scaling</p>
    """, unsafe_allow_html=True)
