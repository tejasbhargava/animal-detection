import argparse
import ast
import os
import csv
from collections import defaultdict


def process(classes, out_dir, yolov8_format):
    """
    Converts Open Images bbox annotations to YOLO format
    """

    # ---- PATHS (EDIT IF NEEDED) ----
    annotations_csv = "train-annotations-bbox.csv"   # path to Open Images CSV
    images_dir = "images"                             # where images are stored

    labels_out = os.path.join(out_dir, "labels")
    images_out = os.path.join(out_dir, "images")

    os.makedirs(labels_out, exist_ok=True)
    os.makedirs(images_out, exist_ok=True)

    # Store annotations per image
    image_to_boxes = defaultdict(list)

    print("[INFO] Reading annotations CSV...")

    with open(annotations_csv, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["LabelName"] not in classes:
                continue

            image_id = row["ImageID"]

            xmin = float(row["XMin"])
            xmax = float(row["XMax"])
            ymin = float(row["YMin"])
            ymax = float(row["YMax"])

            # YOLO format
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            image_to_boxes[image_id].append(
                f"0 {x_center} {y_center} {width} {height}"
            )

    print(f"[INFO] Found {len(image_to_boxes)} images with Monkey")

    # Write YOLO txt files
    for image_id, boxes in image_to_boxes.items():
        label_path = os.path.join(labels_out, f"{image_id}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(boxes))

    print("[DONE] YOLO labels written to:", labels_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--classes', default=['/m/08pbxl'])  # Monkey
    parser.add_argument('--out-dir', default='./data')
    parser.add_argument('--yolov8-format', default=True)

    args = parser.parse_args()

    classes = args.classes
    if type(classes) is str:
        classes = ast.literal_eval(classes)

    out_dir = args.out_dir

    yolov8_format = True if args.yolov8_format in ['T', 'True', 1, '1'] else False

    process(classes, out_dir, yolov8_format)

