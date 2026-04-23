import streamlit as st
import requests
import cv2
import numpy as np

# Page Config
st.set_page_config(
    page_title="Thermal HAR System",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#ff4b4b;
        text-align:center;
    }
    .sub-text {
        text-align:center;
        font-size:18px;
        color:gray;
    }
    .prediction-box {
        background-color:#1e1e1e;
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:28px;
        font-weight:bold;
        color:#00ffcc;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title"> Human Activity Recognition Using Thermal Images</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload exactly 8 frames to predict activity</p>', unsafe_allow_html=True)

st.divider()

#  Upload Section
uploaded_files = st.file_uploader(
    "📂 Upload 8 images",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

#  Preview Section
frames_preview = []

if uploaded_files:
    st.subheader("🖼️ Preview Frames")

    cols = st.columns(4)

    for i, file in enumerate(uploaded_files):

        file.seek(0)   #  FIX: reset pointer

        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

        frames_preview.append(img)

        with cols[i % 4]:
            st.image(img, caption=f"Frame {i+1}", width=150)

#  Validation
if uploaded_files and len(uploaded_files) != 8:
    st.warning(" Please upload exactly 8 images")

#  Prediction Button
if uploaded_files and len(uploaded_files) == 8:

    if st.button(" Predict Activity", use_container_width=True):

        with st.spinner(" Analyzing motion..."):

            files = []

            for f in uploaded_files:
                f.seek(0)   #  IMPORTANT FIX
                content = f.read()

                files.append(
                    ("files", (f.name, content, "image/png"))
                )

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                #  Safe JSON handling
                try:
                    result = response.json()
                except Exception:
                    st.error(" Backend did not return JSON")
                    st.text(response.text)
                    st.stop()

                #  Output
                if "prediction" in result:
                    prediction = result["prediction"]

                    st.markdown(
                        f'<div class="prediction-box">Prediction: {prediction}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error(result)

            except Exception as e:
                st.error(f" Connection Error: {e}")

st.divider()
