# FootballAI
# Football Player Tracking Pipeline

## Overview

This project implements a computer vision pipeline for detecting and tracking football players in a video using YOLO and ByteTrack.

It outputs:

- Annotated video with player IDs
- Average speed estimation
- Heatmaps per player

## Setup

pip install -r requirements.txt

## Run

python main.py

## Pipeline

Video
 → YOLOv8 Player Detection
 → ByteTrack Tracking
 → Speed Estimation
 → Heatmap Generation

## Assumptions

- Players detected using COCO "person" class
- Speed measured in pixel displacement
- Camera assumed static

## Limitations

- Speed not calibrated to meters
- Player occlusion may break tracking
- No team classification

## Improvements

- Homography for real-world speed
- Team classification
- Ball tracking
