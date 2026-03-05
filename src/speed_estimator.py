import numpy as np

class SpeedEstimator:

    def __init__(self, fps):
        self.fps = fps
        self.positions = {}

    def update(self, track_id, center):

        if track_id not in self.positions:
            self.positions[track_id] = []

        self.positions[track_id].append(center)

    def compute_speeds(self):

        speeds = {}

        for tid, points in self.positions.items():

            if len(points) < 2:
                continue

            distances = []

            for i in range(1, len(points)):
                p1 = np.array(points[i-1])
                p2 = np.array(points[i])

                d = np.linalg.norm(p2 - p1)
                distances.append(d)

            avg_pixels_per_frame = np.mean(distances)
            speed = avg_pixels_per_frame * self.fps

            speeds[tid] = float(speed)

        return speeds
