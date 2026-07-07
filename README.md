# 🐾 Animal Face Classification using VGG16

A deep learning image classification project that uses a pretrained **VGG16** model built with **TensorFlow/Keras** to classify animal face images into **three categories**: **Cat**, **Dog**, and **Wild**.

The project includes a modern **Streamlit** web application that allows users to upload an animal face image and receive instant predictions with confidence scores.

---

## 🚀 Live Demo

👉 **(https://8dx6ervsmcinkaxunhcrnp.streamlit.app/)**

---

## ✨ Features

- 🐱 Classifies 3 animal face categories
- 🧠 Transfer Learning with VGG16
- 📷 Upload an image for instant prediction
- 📊 Displays Top-3 predictions with confidence scores
- ⚡ Fast inference using TensorFlow
- 🌐 Interactive Streamlit web application
- 💻 CPU-compatible deployment

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- VGG16
- Streamlit
- NumPy
- Pillow
- Pandas
- Matplotlib

---

## 📂 Project Structure

```text
VGG-Classify-Animal-Faces/
│
├── app.py
├── VGG.h5
├── requirements.txt
├── README.md
└── sample_images/
```

---

## 📊 Dataset

This project was trained using the **Animal Faces (AFHQ)** dataset from Kaggle.

**Dataset Link**

https://www.kaggle.com/datasets/andrewmvd/animal-faces

### Classes

- 🐱 Cat
- 🐶 Dog
- 🦁 Wild

---

## 🧠 Model Architecture

The project uses a pretrained **VGG16** model as the feature extractor for transfer learning.

```text
Input Image (224×224)
        │
Image Preprocessing
        │
Pretrained VGG16
        │
Fully Connected Layers
        │
Softmax
        │
Cat • Dog • Wild
```

---

## 💾 Model Saving

The trained model was saved using TensorFlow's `.h5` format.

```python
model.save("VGG.h5")
```

The saved model is loaded during inference using:

```python
from tensorflow.keras.models import load_model

model = load_model("VGG.h5")
```

---

## 📈 Model Performance

| Metric | Value |
|---------|------:|
| Model | VGG16 |
| Framework | TensorFlow |
| Classes | 3 |
| Task | Animal Face Classification |
| Input Size | 224 × 224 |
| Output | Cat, Dog, Wild |

> Replace this section with your final validation accuracy if available.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/VGG-Classify-Animal-Faces.git

cd VGG-Classify-Animal-Faces
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

- Improve classification accuracy with additional training
- Experiment with EfficientNet and Vision Transformers (ViT)
- Add Grad-CAM visualizations
- Deploy using Docker
- Support batch image predictions

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Mohamed Ahmed**

If you found this project useful, consider giving it a ⭐ on GitHub!
