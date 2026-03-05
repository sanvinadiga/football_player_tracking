import numpy as np
import cv2
import os

class HeatmapGenerator:

    def __init__(self, frame_shape):
        self.maps = {}
        self.h, self.w = frame_shape[:2]

    def update(self, track_id, center):

        if track_id not in self.maps:
            self.maps[track_id] = np.zeros((self.h, self.w))

        x, y = int(center[0]), int(center[1])

        if 0 <= x < self.w and 0 <= y < self.h:
            self.maps[track_id][y, x] += 1

    def save(self, output_dir):

        os.makedirs(output_dir, exist_ok=True)

        for tid, heatmap in self.maps.items():

            normalized = cv2.normalize(
                heatmap,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            heatmap_img = cv2.applyColorMap(
                normalized.astype(np.uint8),
                cv2.COLORMAP_JET
            )

            cv2.imwrite(
                f"{output_dir}/player_{tid}.png",
                heatmap_img
            )
