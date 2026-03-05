import cv2
from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.speed_estimator import SpeedEstimator
from src.heatmap_generator import HeatmapGenerator

class VideoProcessor:

    def __init__(self, input_path, output_path):

        self.cap = cv2.VideoCapture(input_path)

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        width = int(self.cap.get(3))
        height = int(self.cap.get(4))

        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        self.detector = PlayerDetector()
        self.tracker = PlayerTracker()
        self.speed_estimator = SpeedEstimator(fps)
        self.heatmap_generator = HeatmapGenerator((height, width))

    def process(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)

            for x1, y1, x2, y2, track_id in tracks:

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                self.speed_estimator.update(track_id, (cx, cy))
                self.heatmap_generator.update(track_id, (cx, cy))

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

            self.writer.write(frame)

        self.cap.release()
        self.writer.release()

        speeds = self.speed_estimator.compute_speeds()
        self.heatmap_generator.save("output/heatmaps")

        return speeds
