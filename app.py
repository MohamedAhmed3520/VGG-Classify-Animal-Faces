import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Animal Face Classifier",
    page_icon="🐾",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#eef7ff,#dff6ff,#ffffff);
}

#MainMenu,footer,header{
visibility:hidden;
}

.title{
text-align:center;
font-size:48px;
font-weight:bold;
color:#1565c0;
}

.subtitle{
text-align:center;
color:#666;
margin-bottom:25px;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 12px rgba(0,0,0,.12);
}

.stButton>button{
width:100%;
height:50px;
background:#1976d2;
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
}

.terminal{
background:#111;
color:#00ff66;
padding:18px;
border-radius:12px;
font-family:monospace;
white-space:pre-wrap;
}

</style>

<div class="title">
🐾 Animal Face Classifier
</div>

<div class="subtitle">
VGG16 • TensorFlow • 3 Animal Classes
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load():
    return load_model("VGG.h5")

model = load()

# -----------------------------
# Classes
# -----------------------------

class_names = [
    "Cat",
    "Dog",
    "Wild"
]

IMG_SIZE = 224

# -----------------------------
# Layout
# -----------------------------

left, right = st.columns([2,1])

with left:

    uploaded = st.file_uploader(
        "Upload an animal face",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        img = Image.open(uploaded).convert("RGB")

        st.image(
            img,
            use_container_width=True
        )

with right:

    st.markdown("""
<div class="card">

<h3>🤖 Model Information</h3>
<b>Dataset</b><br>
<a href="https://www.kaggle.com/datasets/andrewmvd/animal-faces"
target="_blank">
<br>

<b>Architecture</b><br>
VGG16

<br><br>

<b>Framework</b><br>
TensorFlow



<b>Classes</b><br>

• Cat<br>
• Dog<br>
• Wild



</div>
""", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------

if uploaded and st.button("🚀 Predict"):

    progress = st.progress(0)

    for i in range(100):
        progress.progress(i+1)
        time.sleep(.005)

    progress.empty()

    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = image.img_to_array(img)

    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    top3 = np.argsort(preds)[::-1]

    terminal = st.empty()

    txt = f"""

> MODEL LOADED

> IMAGE PREPROCESSED

> RUNNING VGG16

> RESULT

{class_names[top3[0]]}

> SUCCESS

"""

    cur = ""

    for c in txt:

        cur += c

        terminal.markdown(
            f"<div class='terminal'>{cur}</div>",
            unsafe_allow_html=True
        )

        time.sleep(.002)

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Prediction",
            class_names[top3[0]],
            f"{preds[top3[0]]*100:.2f}%"
        )

        df = pd.DataFrame({

            "Class":[class_names[i] for i in top3],

            "Confidence (%)":[preds[i]*100 for i in top3]

        })

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

    with c2:

        fig, ax = plt.subplots(figsize=(6,3))

        ax.barh(
            [class_names[i] for i in top3][::-1],
            [preds[i]*100 for i in top3][::-1]
        )

        ax.set_xlabel("Confidence (%)")

        st.pyplot(fig)

st.markdown("---")

st.markdown(
"""
<center>

🐾 VGG16 • TensorFlow • Streamlit

</center>
""",
unsafe_allow_html=True
)
