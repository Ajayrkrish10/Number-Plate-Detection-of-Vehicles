# 🚗 Number Plate Detection of Vehicles

An AI-based system to **automatically detect and recognize vehicle number plates** using computer vision. This project utilizes the YOLO object detection model and OCR techniques to detect plates and extract alphanumeric text in real-time, integrated into a Streamlit web application.

---

## 👥 Team Members

- **Ananthan M P**
- **Ajay R Krishnan**
- **Ajay Rajas**
- **Dhanush S Kumar**

---

## 💡 Problem Statement

Manual number plate recognition is slow and error-prone. This project addresses the need for **real-time, automated vehicle number plate detection and OCR** for applications such as traffic control, law enforcement, parking automation, and toll collection.

---

## 📂 Project Structure

.
├── add_missing_data.py # Script to interpolate missing bounding box data
├── yolov5/ # Folder for YOLO training & model files (assumed)
├── app.py # Streamlit web application (if added)
├── data/ # Folder for images and labels
├── test.csv # Original annotated dataset
├── test_interpolated.csv # Cleaned dataset after interpolation
├── README.md # This file
├── requirements.txt # Python dependencies

---

## 🚀 Features

- 📸 **YOLO-based number plate detection**
- 🔤 **OCR integration (Tesseract or EasyOCR)** to extract license numbers
- 🖼️ **Real-time webcam support and image uploads**
- 📈 Detection results with **confidence scores, overlays, and logs**
- 📊 Optional performance dashboard for metrics

---

## 🧪 Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
---

▶️ How to Run
1. Prepare the Dataset
Use YOLO-formatted data (images + .txt annotation files)

To clean or interpolate missing bounding boxes, use:

bash
Copy
Edit
python add_missing_data.py
This will generate test_interpolated.csv.

2. Train the YOLO Model
Assuming you are using YOLOv5:

bash
Copy
Edit
cd yolov5
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
3. Run the Streamlit App (Optional GUI)
bash
Copy
Edit
streamlit run app.py
Upload an image or use the webcam to detect plates and extract text.

---
📊 Evaluation Metrics
Mean Average Precision (mAP)

IoU (Intersection over Union)

OCR Text Accuracy

False Positive / Negative Rate

Inference Time (< 50ms per image)

---
📁 Dataset
[Download Dataset (Google Drive)](https://drive.google.com/file/d/1HAyBtLZGuzHyu2URNE25W5U6nvl15a_G/view)

Images with number plates

YOLO bounding box annotations

Augmentation: rotation, noise, brightness, flip, crop

---
🧬 Typical Usage
👮 Police can automate number plate tracking

🅿️ Parking systems validate vehicles

🚚 Fleet managers can track vehicle movement

🚦 Governments analyze traffic for planning

🏢 Gated communities control entry access

---
📦 Deliverables
✅ Trained YOLO Model

✅ OCR pipeline for plate recognition

✅ Streamlit web app

✅ Cleaned and augmented dataset

✅ Performance report

✅ Deployment instructions

---
🧠 Tech Stack
Python, OpenCV, NumPy

YOLOv5 (PyTorch-based)

EasyOCR / Tesseract

Streamlit for UI

Pandas, Matplotlib for EDA

---
🛠️ Deployment
You can deploy the Streamlit app on:

Local machine

AWS/GCP using Docker

Streamlit Cloud

---
📜 License
For research and educational use only. Not intended for real-world law enforcement unless legally approved.

---

### ✅ `requirements.txt`

```txt
numpy
opencv-python
streamlit
easyocr
pandas
scipy
matplotlib
seaborn
torch>=1.7
