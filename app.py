import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv


# --------------------------------------
# Initialize models & tracker (cached)
# --------------------------------------
@st.cache_resource
def load_models():
    tracker = Sort()
    coco_model = YOLO('yolov8n.pt')
    lp_model = YOLO('license_plate_detector.pt')
    return tracker, coco_model, lp_model


tracker, coco_model, lp_model = load_models()
vehicles = [2, 3, 5, 7]


# --------------------------------------
# Frame processing function
# --------------------------------------
def process_frame(frame, frame_nmr, results):
    # Vehicle detection & tracking
    detections = coco_model(frame)[0]
    filtered = [[x1, y1, x2, y2, score]
                for x1, y1, x2, y2, score, cls in detections.boxes.data.tolist()
                if int(cls) in vehicles]
    tracks = tracker.update(np.asarray(filtered))

    # License plate detection
    lp_dets = lp_model(frame)[0]
    for x1, y1, x2, y2, score, _ in lp_dets.boxes.data.tolist():
        xcar1, ycar1, xcar2, ycar2, track_id = get_car((x1, y1, x2, y2, score, None), tracks)
        if track_id == -1:
            continue
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        text, text_score = read_license_plate(thresh)
        if text is None:
            continue
        results.setdefault(frame_nmr, {})[track_id] = {
            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            'license_plate': {'bbox': [x1, y1, x2, y2], 'text': text,
                              'bbox_score': score, 'text_score': text_score}
        }
    return results


# --------------------------------------
# App UI
# --------------------------------------
st.title("Vehicle & License Plate Detection")

uploaded = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"])

if uploaded is not None:
    # Save to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp_file.name + os.path.splitext(uploaded.name)[1]
    with open(tmp_path, 'wb') as f:
        f.write(uploaded.read())

    ext = os.path.splitext(tmp_path)[1].lower()
    results = {}

    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        frame = cv2.imread(tmp_path)
        results = process_frame(frame, 0, results)
        # Annotate image
        for _, cars in results.items():
            for cid, info in cars.items():
                x1, y1, x2, y2 = map(int, info['car']['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                lx1, ly1, lx2, ly2 = map(int, info['license_plate']['bbox'])
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
                cv2.putText(frame, info['license_plate']['text'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Annotated Image")

    else:
        # Process video
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_tmp = tmp_path + '_out.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_tmp, fourcc, fps, (w, h))

        frame_nmr = 0
        with st.spinner('Processing video...'):
            while True:
                ret, frame = cap.read()
                if not ret: break
                results = process_frame(frame, frame_nmr, results)
                for cid, info in (results.get(frame_nmr) or {}).items():
                    x1, y1, x2, y2 = map(int, info['car']['bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lx1, ly1, lx2, ly2 = map(int, info['license_plate']['bbox'])
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
                    cv2.putText(frame, info['license_plate']['text'], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                writer.write(frame)
                frame_nmr += 1
            cap.release()
            writer.release()
        st.video(out_tmp)

    # Display CSV results
    csv_path = tmp_path + '_results.csv'
    write_csv(results, csv_path)
    df = pd.read_csv(csv_path)
    st.subheader("Detection Results")
    st.dataframe(df)

    # Cleanup temp files
    os.remove(tmp_path)
    if 'out_tmp' in locals(): os.remove(out_tmp)
    os.remove(csv_path)
