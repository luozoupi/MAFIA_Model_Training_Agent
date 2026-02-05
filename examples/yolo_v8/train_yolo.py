from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="coco128.yaml",
        epochs=5,
        imgsz=640,
        project="runs/yolo_v8",
        name="coco128_demo",
    )
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    print("Training complete. Metrics:", metrics)


if __name__ == "__main__":
    main()
