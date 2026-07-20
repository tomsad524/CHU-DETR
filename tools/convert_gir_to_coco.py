#!/usr/bin/env python3
"""
Convert GIR Dataset to COCO format.

GIR is a custom dataset built from RGBT210 video sequences for the CHU-DETR
paper. It contains 5,105 image pairs with 5 object classes.

Expected raw dataset structure (before conversion):
    gir_raw/
    ├── visible/         # visible images (any naming)
    │   ├── img_0001.jpg
    │   ├── img_0002.jpg
    │   └── ...
    ├── thermal/         # infrared images (must match visible names)
    │   ├── img_0001.jpg
    │   ├── img_0002.jpg
    │   └── ...
    └── annotations/     # or a single COCO/CSV annotation file
        ├── train.txt    # one line per image: filename x1 y1 x2 y2 class_id
        └── val.txt

Or if annotations are already in COCO JSON format, place them at:
    gir_raw/annotations/train.json
    gir_raw/annotations/val.json

This script handles both cases and outputs:
    gir_coco/
    ├── annotations/
    │   ├── train.json      # COCO format with file_name_RGB and file_name_IR
    │   └── val.json
    ├── train_RGB/
    ├── train_thermal/
    ├── val_RGB/
    └── val_thermal/

GIR Categories:
    1: person
    2: dog
    3: car
    4: bicycle
    5: motorcycle

Usage:
    python tools/convert_gir_to_coco.py \
        --gir_root /path/to/gir_raw/ \
        --output_root /path/to/gir_coco/ \
        --split all
"""

import argparse
import json
import os
import sys
from pathlib import Path

CATEGORIES = [
    {"id": 1, "name": "person",    "supercategory": "object"},
    {"id": 2, "name": "dog",       "supercategory": "animal"},
    {"id": 3, "name": "car",       "supercategory": "vehicle"},
    {"id": 4, "name": "bicycle",   "supercategory": "vehicle"},
    {"id": 5, "name": "motorcycle","supercategory": "vehicle"},
]


def parse_txt_annotations(txt_file):
    """
    Parse a simple TXT annotation file.

    Expected format (one object per line):
        filename x1 y1 x2 y2 class_id

    Multiple objects in the same image appear as consecutive lines with the
    same filename.

    Returns:
        dict: {filename: [{"bbox": [x,y,w,h], "category_id": id}, ...]}
    """
    annots = {}
    if not os.path.exists(txt_file):
        print(f"  Warning: {txt_file} not found.")
        return annots

    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            fname = parts[0]
            try:
                x1, y1, x2, y2 = map(float, parts[1:5])
                cat_id = int(parts[5])
            except ValueError:
                continue

            # Convert (x1,y1,x2,y2) to (x,y,w,h) COCO bbox format
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue
            if cat_id < 1 or cat_id > 5:
                continue

            if fname not in annots:
                annots[fname] = []
            annots[fname].append({
                "bbox": [x, y, w, h],
                "category_id": cat_id,
            })

    return annots


def build_coco_from_images(gir_root, output_root, split='train',
                           use_symlinks=True):
    """
    Build COCO dataset from raw image directories and annotation files.

    Supports both:
        - TXT annotation files (gir_root/annotations/{split}.txt)
        - Pre-existing COCO JSON (gir_root/annotations/{split}.json)
    """
    gir_root = Path(gir_root)
    output_root = Path(output_root)

    # Check for pre-existing COCO JSON
    json_path = gir_root / "annotations" / f"{split}.json"
    txt_path = gir_root / "annotations" / f"{split}.txt"

    # Output directories
    rgb_dir = output_root / f"{split}_RGB"
    thermal_dir = output_root / f"{split}_thermal"
    ann_dir = output_root / "annotations"
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(thermal_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    if json_path.exists():
        # Already in COCO format — just ensure file_name_RGB/IR keys exist
        print(f"  Loading existing COCO JSON: {json_path}")
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        # Ensure each image entry has both file_name_RGB and file_name_IR
        for img in coco_data.get("images", []):
            if "file_name_RGB" not in img:
                img["file_name_RGB"] = img.get("file_name", "")
            if "file_name_IR" not in img:
                img["file_name_IR"] = img.get("file_name", "")

        # Copy/link images
        vis_src = gir_root / "visible"
        therm_src = gir_root / "thermal"
        for img in coco_data.get("images", []):
            fname = img.get("file_name", "")
            if not fname:
                continue
            vis_in = vis_src / fname
            therm_in = therm_src / fname
            vis_out = rgb_dir / fname
            therm_out = thermal_dir / fname
            if use_symlinks:
                if vis_in.exists() and not vis_out.exists():
                    os.symlink(vis_in.resolve(), vis_out)
                if therm_in.exists() and not therm_out.exists():
                    os.symlink(therm_in.resolve(), therm_out)
            else:
                import shutil
                if vis_in.exists() and not vis_out.exists():
                    shutil.copy2(vis_in, vis_out)
                if therm_in.exists() and not therm_out.exists():
                    shutil.copy2(therm_in, therm_out)

        # Save processed COCO JSON
        out_json = ann_dir / f"{split}.json"
        with open(out_json, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"  {split}: {len(coco_data['images'])} pairs, "
              f"{len(coco_data['annotations'])} boxes -> {out_json}")
        return

    # Build from TXT annotations
    print(f"  Building from TXT annotations: {txt_path}")
    annots = parse_txt_annotations(txt_path)

    vis_src_dir = gir_root / "visible"
    therm_src_dir = gir_root / "thermal"

    # Collect all image files
    vis_files = set()
    if vis_src_dir.exists():
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            vis_files.update(f.name for f in vis_src_dir.glob(ext))

    therm_files = set()
    if therm_src_dir.exists():
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            therm_files.update(f.name for f in therm_src_dir.glob(ext))

    # Only use files that exist in BOTH modalities
    paired_files = sorted(vis_files & therm_files)

    coco_images = []
    coco_annotations = []
    image_id = 1
    ann_id = 1
    total_boxes = 0

    for fname in paired_files:
        vis_path = vis_src_dir / fname
        therm_path = therm_src_dir / fname

        # Copy/link
        vis_out = rgb_dir / fname
        therm_out = thermal_dir / fname
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

        # Image dimensions
        from PIL import Image
        try:
            with Image.open(vis_path) as img:
                width, height = img.size
        except Exception:
            width, height = 640, 512

        img_entry = {
            "id": image_id,
            "file_name_RGB": fname,
            "file_name_IR": fname,
            "width": width,
            "height": height,
        }
        coco_images.append(img_entry)

        # Add annotations for this image
        boxes = annots.get(fname, [])
        for box in boxes:
            x, y, w, h = box["bbox"]
            x = max(0, x); y = max(0, y)
            w = min(w, width - x); h = min(h, height - y)
            if w <= 0 or h <= 0:
                continue
            ann_entry = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": box["category_id"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            coco_annotations.append(ann_entry)
            ann_id += 1
        total_boxes += len(boxes)
        image_id += 1

    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CATEGORIES,
    }

    out_json = ann_dir / f"{split}.json"
    with open(out_json, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"  {split}: {len(coco_images)} pairs, {total_boxes} boxes -> {out_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GIR dataset to COCO format for CHU-DETR"
    )
    parser.add_argument('--gir_root', type=str, required=True,
                        help='Path to raw GIR dataset root.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Path to output COCO-format directory.')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'all'],
                        help='Which split to convert.')
    parser.add_argument('--use_symlinks', action='store_true', default=True,
                        help='Create symlinks instead of copying images.')
    args = parser.parse_args()

    print(f"Converting GIR dataset...")
    print(f"  Source: {args.gir_root}")
    print(f"  Output: {args.output_root}")

    if args.split in ('train', 'all'):
        build_coco_from_images(
            args.gir_root, args.output_root, 'train', args.use_symlinks)
    if args.split in ('val', 'all'):
        build_coco_from_images(
            args.gir_root, args.output_root, 'val', args.use_symlinks)

    print(f"\nDone! To train on GIR:")
    print(f"  python main.py -c config/DINO/DINO_4scale.py \\")
    print(f"      --dataset_file gir_fusion \\")
    print(f"      --coco_path {args.output_root} \\")
    print(f"      --num_classes 6")


if __name__ == '__main__':
    main()
