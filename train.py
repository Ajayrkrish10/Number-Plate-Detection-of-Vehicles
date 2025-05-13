import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune a YOLOv8 model on a license plate dataset."
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Pretrained YOLO model or path to checkpoint."
    )
    parser.add_argument(
        "--data", type=str, default="data/plates.yaml",
        help="Path to data YAML file (e.g., data/plates.yaml)."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for training (pixels)."
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size per GPU."
    )
    parser.add_argument(
        "--project", type=str, default="runs/plate_train",
        help="Output project directory."
    )
    parser.add_argument(
        "--name", type=str, default="yolov8n-plates",
        help="Experiment name (subfolder under project)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint in project/name."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device specifier (e.g., '0', '0,1', 'cpu')."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify data YAML exists
    if not os.path.isfile(args.data):
        print(f"Data YAML file '{args.data}' not found. Please create a YAML with paths to 'images' and 'yolo_labels'.")
        exit(1)

    # Ensure project directory exists
    os.makedirs(args.project, exist_ok=True)

    # Initialize YOLO model
    model = YOLO(args.model)

    # Start training
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        device=args.device
    )

if __name__ == '__main__':
    main()
