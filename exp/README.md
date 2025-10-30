# RT-IRCA Training Documentation

## Overview
RT-IRCA (Real-time Infrared Context Aggregation) is a lightweight model based on Ultralytics YOLO for substation equipment detection.

## Training Workflow

### 1. Pre-train the Teacher Model
```bash
yolo train cfg=pretrain.yaml
```

### 2. Train the Student Model with Knowledge Distillation
```bash
python mlkd.py
```

### 3. Validate the Student Model
```bash
yolo val data=ISED.yaml split=test model=path/to/best.pt
```