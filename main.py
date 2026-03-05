import json
from src.video_processor import VideoProcessor

INPUT_VIDEO = "input/match.mp4"
OUTPUT_VIDEO = "output/annotated_video.mp4"

processor = VideoProcessor(INPUT_VIDEO, OUTPUT_VIDEO)

speeds = processor.process()

with open("output/player_speeds.json", "w") as f:
    json.dump(speeds, f, indent=4)

print("Player Speeds:")

for pid, speed in speeds.items():
    print(f"Player {pid}: {speed:.2f} pixels/sec")
