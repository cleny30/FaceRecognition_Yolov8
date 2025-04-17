from ultralytics import YOLO
import cv2
import os
import pickle
from imutils import paths

def encode_faces(dataset_path: str, encodings_path: str):
    print("[INFO] Quantifying faces...")
    image_paths = list(paths.list_images(dataset_path))

    known_encodings = []
    known_names = []

    if os.path.exists(encodings_path):
        print("[INFO] Loading existing encodings...")
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
            known_encodings = data["encodings"]
            known_names = data["names"]

    # Load YOLOv8 model
    model = YOLO("face_yolov8m.pt")  # Load a pretrained YOLOv8 face detection model

    for (i, image_path) in enumerate(image_paths):
        print(f"[INFO] Processing image {i + 1}/{len(image_paths)}: {image_path}")
        name = image_path.split(os.path.sep)[-2]

        image = cv2.imread(image_path)
        results = model(image, conf=0.5)  # Detect faces using YOLOv8

        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = image[y1:y2, x1:x2]
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Compute facial embeddings (you can use OpenCV or another method here)
                # For simplicity, let's assume we use a simple feature extraction method
                encoding = cv2.dnn.blobFromImage(rgb, 1.0, (96, 96), (104.0, 177.0, 123.0), swapRB=True, crop=False)
                known_encodings.append(encoding.flatten())
                known_names.append(name)

    print("[INFO] Serializing updated encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_path, "wb") as f:
        f.write(pickle.dumps(data))

    return {
        'message': 'File successfully uploaded, extracted, processed, and trained',
        'total_images': len(image_paths),
        'successfully_aligned_images': len(image_paths),
        'classifier_file': encodings_path
    }