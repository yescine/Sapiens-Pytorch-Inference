import torch
import cv2
import os
import argparse
import json
from tqdm import tqdm
from glob import glob
from imread_from_url import imread_from_url

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map, classes, segmentation_to_polygons, polygons_to_mask

from sapiens_inference.cvat import (
    create_cvat_root,
    create_cvat_image,
    add_cvat_polygon,
    write_cvat_xml,
)

def iter_polygons(polygons_obj, segmentation_classes):
    """
    Make segmentation_to_polygons() output iterable as:
        yield (label:str, polygon_pts:list)

    Supports common shapes:
    - dict: {class_id/int or class_name/str: [poly, poly, ...]}
    - list: [ [poly,...] for each class index ]
    """
    if polygons_obj is None:
        return

    # dict form
    if isinstance(polygons_obj, dict):
        for k, poly_list in polygons_obj.items():
            if not poly_list:
                continue

            # resolve label
            label = None
            if isinstance(k, int):
                if 0 <= k < len(segmentation_classes):
                    label = segmentation_classes[k]
                else:
                    label = str(k)
            elif isinstance(k, str):
                # could be "12" or "Torso"
                if k.isdigit():
                    ki = int(k)
                    label = (
                        segmentation_classes[ki]
                        if 0 <= ki < len(segmentation_classes)
                        else k
                    )
                else:
                    label = k
            else:
                label = str(k)

            # poly_list expected: list of polygons
            if isinstance(poly_list, list):
                for poly in poly_list:
                    if poly is None:
                        continue
                    yield (label, poly)
            else:
                # edge case: single polygon
                yield (label, poly_list)
        return

    # list form aligned with classes indices
    if isinstance(polygons_obj, list):
        for idx, poly_list in enumerate(polygons_obj):
            if not poly_list:
                continue
            label = (
                segmentation_classes[idx]
                if 0 <= idx < len(segmentation_classes)
                else str(idx)
            )
            if isinstance(poly_list, list):
                for poly in poly_list:
                    if poly is None:
                        continue
                    yield (label, poly)
            else:
                yield (label, poly_list)
        return

    # unknown structure -> do nothing
    return

def main():
    parser = argparse.ArgumentParser(description="Sapiens Image Segmentation")
    parser.add_argument("--url", type=str, help="URL of the image to segment")
    parser.add_argument("--img_dir", type=str, help="Directory containing images to segment")
    parser.add_argument("--out_dir", type=str, default="./runs", help="Output directory for segmented images")
    parser.add_argument("--model", type=str, default="1b", choices=["03b", "06b", "1b"], help="Model size: 03b, 06b, or 1b")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type for inference")
    parser.add_argument("--save", action="store_true", default=True, help="Save the segmented image (default: True)")
    parser.add_argument("--no-save", action="store_false", dest="save", help="Do not save the segmented image")
    parser.add_argument("--class_name", type=str, help="Class name for the images (falls back to parent dir if not provided)")
    parser.add_argument("--classes-json", type=str, default="./data/classes.json", help="Path to classes JSON file")

    # ✅ CVAT export
    parser.add_argument("--cvat", action="store_true", default=True, help="Generate CVAT annotations.xml (default: True)")
    parser.add_argument("--no-cvat", action="store_false", dest="cvat", help="Disable CVAT export")
    parser.add_argument("--cvat-out", type=str, default=None, help="Path to CVAT annotations.xml (default: out_dir/annotations.xml)")
    parser.add_argument("--cvat-clamp", action="store_true", default=True, help="Clamp polygon points to image bounds (default: True)")
    parser.add_argument("--no-cvat-clamp", action="store_false", dest="cvat_clamp", help="Do not clamp polygon points")

    args = parser.parse_args()

    # Map model size
    model_map = {
        "03b": SapiensSegmentationType.SEGMENTATION_03B,
        "06b": SapiensSegmentationType.SEGMENTATION_06B,
        "1b": SapiensSegmentationType.SEGMENTATION_1B,
    }

    # Map dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    dtype = dtype_map[args.dtype]
    model_type = model_map[args.model]

    estimator = SapiensSegmentation(model_type, dtype=dtype)

    segmentation_classes = classes
    if args.classes_json and os.path.exists(args.classes_json):
        with open(args.classes_json, "r", encoding="utf-8") as f:
            segmentation_classes = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # CVAT init (single XML for all images)
    cvat_root = create_cvat_root()
    cvat_image_id = 0

    images_to_process = []
    if args.url:
        images_to_process.append(("url", args.url))

    if args.img_dir:
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        for ext in extensions:
            images_to_process.extend([("file", f) for f in glob(os.path.join(args.img_dir, ext))])

    if not args.url and not args.img_dir:
        images_to_process.append(("url", "https://upload.wikimedia.org/wikipedia/commons/5/5b/Jogging_with_dog_at_Carcavelos_Beach.jpg"))

    pbar = tqdm(images_to_process, desc="Processing images")
    for source_type, path in pbar:
        if source_type == "url":
            img = imread_from_url(path)
            filename = os.path.basename(path).split("?")[0]
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                filename += ".png"
        else:
            img = cv2.imread(path)
            filename = os.path.basename(path)

        pbar.set_description(f"Processing {filename}")

        if img is None:
            pbar.write(f"Failed to load image: {path}")
            continue

        image_class = args.class_name if args.class_name else (
            os.path.basename(os.path.dirname(path)) if source_type == "file" else "unknown"
        )

        segmentation_map = estimator(img)

        polygons = segmentation_to_polygons(segmentation_map)

        segmentation_image = draw_segmentation_map(segmentation_map)
        combined = cv2.addWeighted(img, 0.5, segmentation_image, 0.7, 0)

        out_path = os.path.join(args.out_dir, filename)
        if args.save:
            cv2.imwrite(out_path, combined)

        # Save JSON (your original behavior)
        json_data = {
            "filename": filename,
            "class": image_class,
            "shape": img.shape[:2],
            "model": f"sapiens-{args.model}",
            "dtype": args.dtype,
            "classes": segmentation_classes,
            "polygons": polygons,
        }
        json_path = os.path.join(args.out_dir, os.path.splitext(filename)[0] + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        # ✅ Add to CVAT XML
        if args.cvat:
            h, w = img.shape[:2]
            image_el = create_cvat_image(
                cvat_root,
                cvat_image_id,
                filename,  # note: name in CVAT should match your dataset image name
                w,
                h,
            )
            cvat_image_id += 1

            added = 0
            for label, poly in iter_polygons(polygons, segmentation_classes):
                ok = add_cvat_polygon(
                    image_el,
                    label,
                    poly,
                    width=w,
                    height=h,
                    clamp=args.cvat_clamp,
                    occluded=0,
                    source="manual",
                )
                if ok:
                    added += 1

            # optional: quick visibility
            # print(f"  CVAT polygons added: {added}")

        if len(images_to_process) == 1:
            cv2.namedWindow("Segmentation Map", cv2.WINDOW_NORMAL)
            cv2.imshow("Segmentation Map", combined)
            cv2.waitKey(0)

    # Write CVAT once at the end
    if args.cvat:
        cvat_out = args.cvat_out or os.path.join(args.out_dir, "annotations.xml")
        out_xml = write_cvat_xml(cvat_root, cvat_out)
        print(f"CVAT → {out_xml}")


if __name__ == "__main__":
    main()