from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # class 0 = person in COCO
            if cls == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])

        return detections
