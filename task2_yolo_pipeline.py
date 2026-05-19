import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


Point = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 2 pipeline: YOLOv8 train / validate / track with line counting."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a YOLOv8 model.")
    train_parser.add_argument("--data", type=str, required=True, help="Path to data yaml.")
    train_parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model.")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--project", type=str, default="runs/task2")
    train_parser.add_argument("--name", type=str, default="train")
    train_parser.add_argument("--device", type=str, default=None)

    val_parser = subparsers.add_parser("val", help="Validate a trained YOLOv8 model.")
    val_parser.add_argument("--data", type=str, required=True, help="Path to data yaml.")
    val_parser.add_argument("--weights", type=str, required=True, help="Path to model weights.")
    val_parser.add_argument("--imgsz", type=int, default=640)
    val_parser.add_argument("--device", type=str, default=None)

    track_parser = subparsers.add_parser(
        "track", help="Run tracking on a video and count line crossings."
    )
    track_parser.add_argument("--weights", type=str, required=True, help="Path to model weights.")
    track_parser.add_argument("--source", type=str, required=True, help="Input video path.")
    track_parser.add_argument("--output", type=str, default="outputs/tracked_output.mp4")
    track_parser.add_argument(
        "--line",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="Virtual counting line: x1 y1 x2 y2",
    )
    track_parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    track_parser.add_argument("--conf", type=float, default=0.25)
    track_parser.add_argument("--iou", type=float, default=0.5)
    track_parser.add_argument("--imgsz", type=int, default=640)
    track_parser.add_argument("--device", type=str, default=None)
    track_parser.add_argument("--show-labels", action="store_true")

    return parser.parse_args()


def train_model(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        pretrained=True,
        plots=True,
    )


def validate_model(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)
    print(metrics)


def side_of_line(point: Point, line_start: Point, line_end: Point) -> float:
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def box_center(box: List[float]) -> Point:
    x1, y1, x2, y2 = box[:4]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def track_and_count(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    line_start = (args.line[0], args.line[1])
    line_end = (args.line[2], args.line[3])
    source = args.source
    output_path = Path(args.output)
    ensure_parent_dir(output_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    last_side: Dict[int, float] = {}
    counted_ids = set()
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        annotated = frame.copy()
        cv2.line(annotated, line_start, line_end, (0, 255, 255), 2)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
            ids = (
                [int(x) for x in boxes.id.cpu().tolist()]
                if boxes.id is not None
                else [None] * len(xyxy)
            )
            classes = (
                [int(x) for x in boxes.cls.cpu().tolist()]
                if boxes.cls is not None
                else [None] * len(xyxy)
            )
            names = results[0].names

            for box, track_id, cls_id in zip(xyxy, ids, classes):
                if track_id is None:
                    continue

                center = box_center(box)
                current_side = side_of_line(center, line_start, line_end)

                if track_id in last_side:
                    previous_side = last_side[track_id]
                    crossed = previous_side * current_side < 0
                    if crossed and track_id not in counted_ids:
                        counted_ids.add(track_id)

                last_side[track_id] = current_side

                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.circle(annotated, center, 4, (0, 0, 255), -1)

                if args.show_labels:
                    class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    label = f"ID {track_id} | {class_name}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

        cv2.putText(
            annotated,
            f"Cross Count: {len(counted_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Frame: {frame_index}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(annotated)
        frame_index += 1

    cap.release()
    writer.release()
    print(f"Saved tracked video to: {output_path}")
    print(f"Final cross count: {len(counted_ids)}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_model(args)
    elif args.command == "val":
        validate_model(args)
    elif args.command == "track":
        track_and_count(args)


if __name__ == "__main__":
    main()
