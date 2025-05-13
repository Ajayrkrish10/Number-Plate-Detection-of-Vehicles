import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv


def process_frame(frame, tracker, coco_model, lp_model, frame_nmr, results_dict, vehicles):
    # Detect vehicles
    detections = coco_model(frame)[0]
    filtered = []
    for x1, y1, x2, y2, score, cls in detections.boxes.data.tolist():
        if int(cls) in vehicles:
            filtered.append([x1, y1, x2, y2, score])

    # Update tracker
    tracks = tracker.update(np.asarray(filtered))

    # Detect license plates
    lp_dets = lp_model(frame)[0]
    for x1, y1, x2, y2, score, _ in lp_dets.boxes.data.tolist():
        # Assign plate to car
        car = get_car((x1, y1, x2, y2, score, None), tracks)
        if car[4] == -1:
            continue
        xcar1, ycar1, xcar2, ycar2, track_id = car

        # Crop and preprocess LP
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        # OCR
        text, text_score = read_license_plate(thresh)
        if text is None:
            continue

        # Save result
        results_dict.setdefault(frame_nmr, {})[track_id] = {
            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            'license_plate': {
                'bbox': [x1, y1, x2, y2],
                'text': text,
                'bbox_score': score,
                'text_score': text_score
            }
        }
    return results_dict


def annotate_and_save_image(frame, results, output_path):
    # Draw boxes and labels on a single image
    for frame_nmr, cars in results.items():
        for cid, info in cars.items():
            x1, y1, x2, y2 = map(int, info['car']['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lx1, ly1, lx2, ly2 = map(int, info['license_plate']['bbox'])
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
            cv2.putText(frame, info['license_plate']['text'],
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imwrite(output_path, frame)
    print(f"Annotated image saved to {output_path}")


def process_video(input_path, output_path, tracker, coco_model, lp_model, vehicles):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    results = {}
    frame_nmr = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = process_frame(frame, tracker, coco_model, lp_model, frame_nmr, results, vehicles)
        # draw on frame
        for cid, info in (results.get(frame_nmr) or {}).items():
            x1, y1, x2, y2 = map(int, info['car']['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lx1, ly1, lx2, ly2 = map(int, info['license_plate']['bbox'])
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
            cv2.putText(frame, info['license_plate']['text'],
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        writer.write(frame)
        frame_nmr += 1

    cap.release()
    writer.release()

    # Export CSV
    csv_path = 'results.csv'
    write_csv(results, csv_path)
    print(f"Processed video saved to {output_path}, CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Process an image or video for vehicle and license plate detection.")
    parser.add_argument('input', nargs='?', default=None, help="Path to input image/video; if omitted, uses webcam.")
    parser.add_argument('-o', '--output', default=None, help="Path to save annotated output")
    args = parser.parse_args()

    inp = args.input
    use_webcam = False
    if inp is None:
        use_webcam = True
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.isfile(inp):
            print(f"Input '{inp}' not found.")
            return
        cap = cv2.VideoCapture(inp)

    out = args.output
    if out is None:
        if use_webcam:
            out = 'webcam_out.mp4'
        else:
            base, ext = os.path.splitext(inp)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                out = base + '_annotated' + ext
            else:
                out = base + '_out.mp4'

    tracker = Sort()
    coco_model = YOLO('yolov8n.pt')
    lp_model = YOLO('license_plate_detector.pt')
    vehicles = [2, 3, 5, 7]

    # Single image mode
    if not use_webcam and os.path.splitext(inp)[1].lower() in ['.jpg','.jpeg','.png','.bmp']:
        frame = cv2.imread(inp)
        results = process_frame(frame, tracker, coco_model, lp_model, 0, {}, vehicles)
        annotate_and_save_image(frame, results, out)
    else:
        # Video or webcam mode
        if not use_webcam:
            process_video(inp, out, tracker, coco_model, lp_model, vehicles)
        else:
            process_video(0, out, tracker, coco_model, lp_model, vehicles)

if __name__ == '__main__':
    main()