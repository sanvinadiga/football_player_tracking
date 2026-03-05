import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

class PlayerTracker:
    def __init__(self):

        self.tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )

    def update(self, detections):

        if len(detections) == 0:
            return []

        detections = np.array(detections)

        tracks = self.tracker.update(
            detections[:, :4],
            detections[:, 4],
            np.zeros(len(detections))
        )

        results = []

        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            track_id = t.track_id
            results.append([x1, y1, x2, y2, track_id])

        return results
