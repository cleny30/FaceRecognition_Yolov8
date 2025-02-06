from ultralytics import YOLO
import cv2
import pickle
import numpy as np

def recognize_faces_in_image(encodings_path: str, image_path: str):
    print("[INFO] Loading encodings...")
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"status": "error", "message": "Failed to load image."}

    # Load YOLOv8 face detection model
    model = YOLO("face_yolov8n.pt")
    results = model(image)  # Detect faces using YOLOv8

    recognized_faces = []

    # Loop over detected faces
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        if len(boxes) == 0:
            print("[INFO] No faces detected.")
            return recognized_faces

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image[y1:y2, x1:x2]

            # Resize face to 96x96 for feature extraction
            face_resized = cv2.resize(face, (96, 96))
            rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Compute facial embeddings (using a placeholder method)
            encoding = cv2.dnn.blobFromImage(rgb, 1.0, (96, 96), (104.0, 177.0, 123.0), swapRB=True, crop=False)
            encoding = encoding.flatten()

            # Normalize the encoding
            encoding = encoding / np.linalg.norm(encoding)

            # Compare the encoding with known encodings using cosine similarity
            similarities = [np.dot(encoding, known_enc) / (np.linalg.norm(encoding) * np.linalg.norm(known_enc)) for known_enc in data["encodings"]]
            max_similarity = max(similarities)
            max_index = np.argmax(similarities)

            if max_similarity > 0.8:  # Adjust threshold as needed
                name = data["names"][max_index]
                accuracy = float(max_similarity * 100)  # Use similarity directly as accuracy
            else:
                name = "Unknown"
                accuracy = 0.0

            # Thêm kết quả vào danh sách
            recognized_faces.append({
                "detected": name,
                "accuracy": round(float(accuracy), 2)  # Đảm bảo accuracy là kiểu float
            })

    return recognized_faces