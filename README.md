# 🐾 Cat vs Dog Image Classifier (EfficientNetB0)

A professional web-based image classification application built with **Streamlit** and **TensorFlow**. This app allows users to predict whether an image contains a cat or a dog using a high-precision EfficientNetB0 backbone trained on a custom dataset.

---
🚀 **[Live Demo](https://cat-vs-dog-classification-model-07.streamlit.app/)**

## 📊 Model Information & Architecture
The model uses transfer learning based on the EfficientNetB0 architecture, optimized for binary classification.

| Metric | Value |
| :--- | :--- |
| **Architecture** | EfficientNetB0 |
| **Total Parameters** | 5,330,060 (20.33 MB) |
| **Trainable Params** | 1,279,065 (4.88 MB) |
| **Non-trainable Params** | 4,050,995 (15.45 MB) |
| **Input Resolution** | 224 x 224 pixels |

---

## 📂 Project Structure

```text
cat-vs-dog/
├── dataset/
│   ├── train/
│   │   ├── cat/             # Training cat images
│   │   └── dog/             # Training dog images
│   └── test/
│       ├── cat/             # Testing cat images
│       └── dog/             # Testing dog images
├── examples/                # Quick-test images for the App
│   ├── cat1.jpg
│   ├── cat2.jpg
│   ├── dog1.jpg
│   └── dog2.jpg
├── app.py                   # Streamlit Web Application
├── main_notebook.ipynb      # Model Training Notebook
├── my_efficientnet_model.h5   # Trained Model Weights
├── requirements.txt         # Library Dependencies
└── .gitignore               # Files excluded from Git

## How to Run the App

Copy and run the following commands:

```bash
git clone https://github.com/THE-NIKHIL07/cat-vs-dog-classification-model.git

pip install -r requirements.txt
streamlit run app.py
