# Task 2: Scene Object Detection and Multi-Object Tracking

This repository contains my implementation for **Task 2** of the course **Deep Learning and Spatial Intelligence**.

## Task Description

The goal of this task is to:

1. Fine-tune a YOLOv8-based object detector on the **Road Vehicle Images Dataset**
2. Run object detection and multi-object tracking on a 10-30 second traffic video
3. Analyze occlusion and ID switching in consecutive frames
4. Implement a simple line-crossing counting function based on tracking IDs

## Environment

- Python 3.x
- Google Colab with GPU is recommended

Install dependencies:

```bash
pip install ultralytics opencv-python
```

## Dataset

The dataset is organized in YOLO detection format:

- `trafic_data/train/images`
- `trafic_data/train/labels`
- `trafic_data/valid/images`
- `trafic_data/valid/labels`
- `trafic_data/data_1.yaml`

## Main Script

The main script is:

- `task2_yolo_pipeline.py`

It supports:

- training
- validation
- video tracking
- line-crossing counting

## Training

Example command:

```bash
python task2_yolo_pipeline.py train \
  --data trafic_data/data_1.yaml \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 16 \
  --project runs/task2 \
  --name exp1
```

## Validation

Validation is automatically performed during training by Ultralytics YOLOv8.

If needed, validation can also be run manually:

```bash
python task2_yolo_pipeline.py val \
  --data trafic_data/data_1.yaml \
  --weights /content/runs/detect/runs/task2/exp1/weights/best.pt
```

## Tracking and Line-Crossing Counting

Example command:

```bash
python task2_yolo_pipeline.py track \
  --weights /content/runs/detect/runs/task2/exp1/weights/best.pt \
  --source test_video.mp4 \
  --output outputs/tracked_output.mp4 \
  --line 200 350 1000 350 \
  --show-labels
```

This command outputs:

- bounding boxes
- object classes
- tracking IDs
- line-crossing count

## Results

- Detector: YOLOv8n
- Training set: 2704 images
- Validation set: 300 images
- Number of classes: 21
- Final validation result: `mAP50 = 0.419`, `mAP50-95 = 0.259`
- Final line-crossing count in the test video: `7`

## Model Weights

Download link:

- `[Please paste your model weight link here]`

## Notes

- The training process was completed in Google Colab using GPU.
- The test video and tracking result were used for occlusion / ID-switch analysis in the report.

