# YOLOv8 Example Workflow

This example demonstrates a simple YOLOv8 finetuning run on the `coco128` dataset.
It is intentionally small so you can validate the MCP runner end-to-end.

## Prerequisites

```bash
pip install ultralytics
```

## Run

```bash
python3 examples/yolo_v8/train_yolo.py
```

## Notes

- The script uses the `yolov8n.pt` checkpoint and the built-in `coco128` dataset.
- Results will be written to `runs/yolo_v8/coco128_demo`.
