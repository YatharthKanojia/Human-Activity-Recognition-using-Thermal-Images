# 🔥 Thermal Human Activity Recognition (HAR)

A deep learning-based system to recognize human activities using thermal imagery.
The system classifies activities such as **Walking, Running, Standing, and Falling** using a hybrid CNN + Transformer model.

---

## 🚀 Features

* 🎥 Activity recognition using **thermal image sequences**
* 🧠 Hybrid model: **Optical Flow + CNN + Transformer**
* ⚡ FastAPI backend for real-time inference
* 🎨 Streamlit frontend with:

  * Image upload
  * Video/webcam support (optional)
  * Live prediction display
* 📊 Motion-based feature extraction using optical flow

---

## 🧠 Model Architecture

1. **Optical Flow Extraction**

   * Captures motion between consecutive frames

2. **CNN (FlowCNN)**

   * Extracts spatial features from flow maps

3. **Transformer**

   * Learns temporal dependencies across frame sequences

---

## 📂 Project Structure

```
mjrpr/
│
├── model.py          # Model architecture (CNN + Transformer)
├── utils.py          # Preprocessing + optical flow functions
├── predict.py        # Model loading + inference logic
├── app.py            # FastAPI backend (API endpoints)
├── frontend.py       # Streamlit frontend UI
├── HAR_final_model.pth  # Trained model weights
└── README.md
```

---

## ⚙️ Installation

```bash
pip install fastapi uvicorn streamlit torch opencv-python numpy
```

---

## ▶️ How to Run

### 1️⃣ Start Backend

```bash
uvicorn app:app --reload
```

👉 Runs at: http://127.0.0.1:8000

---

### 2️⃣ Start Frontend

```bash
streamlit run frontend.py
```

---

## 🎯 Usage

1. Upload **exactly 8 images** (thermal frames or grayscale)
2. Click **Predict**
3. View predicted activity:

   * Walking
   * Running
   * Standing
   * Falling

---

## 🔗 API Endpoint

### POST `/predict`

* Input: 8 images
* Output:

```json
{
  "prediction": "Walking"
}
```

---

## ⚠️ Limitations

* Model trained on **thermal dataset**, may not generalize to RGB webcam input
* Requires **fixed number of frames (8)**
* Optical flow is sensitive to motion speed

---

## 🚀 Future Improvements

* 🎥 Real-time video-based prediction
* 📸 Webcam-based continuous detection
* 🔄 Domain adaptation for RGB input
* ☁️ Cloud deployment

---

## 👨‍💻 Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Model:** PyTorch
* **Image Processing:** OpenCV, NumPy

---

## 💡 Applications

* Night-time surveillance
* Smart security systems
* Fall detection for elderly care
* Privacy-preserving monitoring

---

## 📌 Author

Yatharth Kanojia
