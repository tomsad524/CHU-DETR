#!/usr/bin/env python3
"""
Convert KAIST Multispectral Pedestrian Dataset to COCO format.

KAIST raw dataset structure:
    kaist_root/
    ├── images/          # contains set00 ~ set11
    │   ├── set00/
    │   │   ├── V000/    # visible images: lwir/I00000.jpg -> visible/I00000.jpg
    │   │   │   ├── visible/
    │   │   │   └── lwir/
    │   │   └── ...
    │   └── ...
    └── annotations/
        └── annotations-sanitized/   # improved annotations

This script converts the dataset to:
    kaist_coco/
    ├── annotations/
    │   ├── train.json    # with "file_name_RGB" and "file_name_IR" keys
    │   └── test.json
    ├── train_RGB/        # symlinks or copies of visible images
    ├── train_thermal/    # symlinks or copies of thermal images
    ├── test_RGB/
    └── test_thermal/

COCO annotation format:
    Each image entry includes:
        {
            "id": image_id,
            "file_name_RGB": "set00_V000_I00001.jpg",
            "file_name_IR": "set00_V000_I00001.jpg",
            "width": 640,
            "height": 512
        }
    Each annotation entry includes:
        {
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,   # "person"
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        }

Usage:
    python tools/convert_kaist_to_coco.py \
        --kaist_root /path/to/kaist/ \
        --output_root /path/to/kaist_coco/ \
        --use_improved_annotations
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Training/validation split from the paper:
# Training:  sets 00-05 (daytime) used for training -> ~7,601 pairs
# Testing:   sets 06-11, sampled frames -> ~2,252 pairs (day + night)
#
# Standard KAIST split (Zhang et al. 2019, improved annotations):
#   train: set00_V000, set00_V001, set01_V000, set02_V000, set02_V001,
#          set03_V000, set03_V001, set04_V000, set04_V001, set05_V000, set05_V001
#   test:  set06_V000, set06_V001, set06_V002, set07_V000, set07_V001,
#          set08_V000, set08_V001, set09_V000, set09_V001, set10_V000,
#          set10_V001, set11_V000, set11_V001

# KAIST categories (pedestrian detection):
CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
]


def parse_kaist_filename(filename):
    """
    Parse a KAIST-style filename like 'set00_V000_I00001.jpg'
    Returns (set_id, video_id, frame_id).
    """
    base = Path(filename).stem  # remove .jpg
    parts = base.split('_')
    if len(parts) != 3:
        return None, None, None
    set_str = parts[0]     # e.g. "set00"
    vid_str = parts[1]     # e.g. "V000"
    img_str = parts[2]     # e.g. "I00001"
    return set_str, vid_str, img_str


def load_annotations(ann_file, use_improved=True):
    """
    Load KAIST annotation file.

    KAIST annotations format (per line per frame):
        frame_name all_object_count obj1_x obj1_y obj1_w obj1_h ...
    """
    annotations = {}
    if not os.path.exists(ann_file):
        print(f"Warning: annotation file {ann_file} not found, skipping.")
        return annotations

    with open(ann_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_name = parts[0]
            try:
                num_objs = int(parts[1])
            except ValueError:
                continue

            boxes = []
            for i in range(num_objs):
                idx = 2 + i * 4
                if idx + 3 >= len(parts):
                    break
                try:
                    x, y, w, h = map(float, parts[idx:idx + 4])
                    if w > 0 and h > 0:
                        boxes.append([x, y, w, h])
                except ValueError:
                    continue
            annotations[frame_name] = boxes

    return annotations


def build_coco_dataset(kaist_root, output_root, split='train',
                       use_improved=True, use_symlinks=True):
    """
    Build COCO-format dataset from raw KAIST files.

    Args:
        kaist_root: Path to raw KAIST directory.
        output_root: Path to output COCO-format directory.
        split: 'train' or 'test'.
        use_improved: Use improved annotations (Zhang et al. 2019).
        use_symlinks: Create symlinks instead of copying images.
    """
    kaist_root = Path(kaist_root)
    output_root = Path(output_root)

    # Define train/test splits
    train_sets = [
        ("set00", "V000"), ("set00", "V001"),
        ("set01", "V000"), ("set01", "V001"),
        ("set02", "V000"), ("set02", "V001"),
        ("set03", "V000"), ("set03", "V001"),
        ("set04", "V000"), ("set04", "V001"),
        ("set05", "V000"), ("set05", "V001"),
    ]
    test_sets = [
        ("set06", "V000"), ("set06", "V001"), ("set06", "V002"),
        ("set07", "V000"), ("set07", "V001"),
        ("set08", "V000"), ("set08", "V001"),
        ("set09", "V000"), ("set09", "V001"),
        ("set10", "V000"), ("set10", "V001"),
        ("set11", "V000"), ("set11", "V001"),
    ]

    video_sets = train_sets if split == 'train' else test_sets

    # Output directories
    rgb_dir = output_root / f"{split}_RGB"
    thermal_dir = output_root / f"{split}_thermal"
    ann_dir = output_root / "annotations"
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(thermal_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    coco_images = []
    coco_annotations = []
    image_id = 1
    ann_id = 1

    total_pairs = 0
    total_boxes = 0

    for set_name, vid_name in video_sets:
        visible_dir = kaist_root / set_name / vid_name / "visible"
        lwir_dir = kaist_root / set_name / vid_name / "lwir"
        # Improved annotations path
        ann_path = (kaist_root / "annotations" / "annotations-sanitized" /
                    set_name / vid_name)

        if not visible_dir.exists() or not lwir_dir.exists():
            print(f"  Warning: {set_name}/{vid_name} images not found, skipping.")
            continue

        # Load annotations
        ann_file = ann_path / "annotations.txt" if use_improved else None
        ann_data = {}
        if ann_file and ann_file.exists():
            ann_data = load_annotations(ann_file, use_improved)

        # Get visible images (they name both modalities the same)
        visible_images = sorted(visible_dir.glob("*.jpg"))
        for vis_path in visible_images:
            frame_name = f"{set_name}/{vid_name}/visible/{vis_path.name}"
            # Corresponding thermal image
            therm_path = lwir_dir / vis_path.name
            if not therm_path.exists():
                continue

            # Create output filenames (unique across dataset)
            out_name = f"{set_name}_{vid_name}_{vis_path.stem}.jpg"

            # Copy or symlink
            vis_out = rgb_dir / out_name
            therm_out = thermal_dir / out_name

            if use_symlinks:
                if not vis_out.exists():
                    os.symlink(vis_path.resolve(), vis_out)
                if not therm_out.exists():
                    os.symlink(therm_path.resolve(), therm_out)
            else:
                import shutil
                if not vis_out.exists():
                    shutil.copy2(vis_path, vis_out)
                if not therm_out.exists():
                    shutil.copy2(therm_path, therm_out)

            # Get image dimensions from the visible image
            from PIL import Image
            try:
                with Image.open(vis_path) as img:
                    width, height = img.size
            except Exception:
                print(f"  Warning: cannot read {vis_path}, using default size.")
                width, height = 640, 512

            # Add image entry to COCO dict
            img_entry = {
                "id": image_id,
                "file_name_RGB": out_name,
                "file_name_IR": out_name,
                "width": width,
                "height": height,
                "set_name": set_name,
                "video_name": vid_name,
            }
            coco_images.append(img_entry)

            # Add annotations
            # Try multiple annotation lookup formats
            ann_key_candidates = [
                f"{set_name}/{vid_name}/visible/{vis_path.stem}.txt",
                vis_path.name,
                vis_path.stem,
            ]
            boxes = []
            for key in ann_key_candidates:
                if key in ann_data:
                    boxes = ann_data[key]
                    break

            for box in boxes:
                x, y, w, h = box
                # Clamp to image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                if w <= 0 or h <= 0:
                    continue

                ann_entry = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # person
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
                coco_annotations.append(ann_entry)
                ann_id += 1

            total_pairs += 1
            total_boxes += len(boxes)
            image_id += 1

    # Write COCO JSON
    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CATEGORIES,
    }

    ann_filename = "train.json" if split == "train" else "test.json"
    ann_path = ann_dir / ann_filename
    with open(ann_path, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"  {split} split: {total_pairs} image pairs, {total_boxes} boxes")
    print(f"  Saved to {ann_path}")

    return total_pairs, total_boxes


def main():
    parser = argparse.ArgumentParser(
        description="Convert KAIST dataset to COCO format for CHU-DETR"
    )
    parser.add_argument('--kaist_root', type=str, required=True,
                        help='Path to raw KAIST dataset root.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Path to output COCO-format directory.')
    parser.add_argument('--use_improved_annotations', action='store_true',
                        default=True,
                        help='Use improved annotations (Zhang et al. 2019).')
    parser.add_argument('--use_symlinks', action='store_true', default=True,
                        help='Create symlinks instead of copying images.')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'test', 'all'],
                        help='Which split to convert.')
    args = parser.parse_args()

    print(f"Converting KAIST dataset...")
    print(f"  Source: {args.kaist_root}")
    print(f"  Output: {args.output_root}")

    if args.split in ('train', 'all'):
        build_coco_dataset(args.kaist_root, args.output_root, 'train',
                          args.use_improved_annotations, args.use_symlinks)
    if args.split in ('test', 'all'):
        build_coco_dataset(args.kaist_root, args.output_root, 'test',
                          args.use_improved_annotations, args.use_symlinks)

    print(f"\nDone! To train on KAIST:")
    print(f"  python main.py -c config/DINO/DINO_4scale.py \\")
    print(f"      --dataset_file kaist_fusion \\")
    print(f"      --coco_path {args.output_root} \\")
    print(f"      --num_classes 2")


if __name__ == '__main__':
    main()
