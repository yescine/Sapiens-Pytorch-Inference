import torch
import cv2
import os
import argparse
import json
from glob import glob
from imread_from_url import imread_from_url

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map, classes, segmentation_to_polygons, polygons_to_mask

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
    if os.path.exists(args.classes_json):
        with open(args.classes_json, 'r') as f:
            segmentation_classes = json.load(f)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    images_to_process = []
    if args.url:
        images_to_process.append(("url", args.url))
    
    if args.img_dir:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            images_to_process.extend([("file", f) for f in glob(os.path.join(args.img_dir, ext))])

    # If no URL or directory provided, and the script is run without arguments, use the default URL
    if not args.url and not args.img_dir:
        images_to_process.append(("url", "https://upload.wikimedia.org/wikipedia/commons/5/5b/Jogging_with_dog_at_Carcavelos_Beach.jpg"))

    for source_type, path in images_to_process:
        if source_type == "url":
            img = imread_from_url(path)
            filename = os.path.basename(path).split('?')[0]
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                filename += ".png"
        else:
            img = cv2.imread(path)
            filename = os.path.basename(path)

        if img is None:
            print(f"Failed to load image: {path}")
            continue

        image_class = args.class_name if args.class_name else (os.path.basename(os.path.dirname(path)) if source_type == "file" else "unknown")

        print(f"Processing {path}...")
        segmentation_map = estimator(img)

        polygons = segmentation_to_polygons(segmentation_map)
        
        # reconstructed_mask = polygons_to_mask(
        #     segmentation_map.shape,
        #     polygons
        # )
        # polygon_segmentation_img = draw_segmentation_map(reconstructed_mask)
        # combined = cv2.addWeighted(img, 0.5, polygon_segmentation_img, 0.7, 0)

        segmentation_image = draw_segmentation_map(segmentation_map)
        combined = cv2.addWeighted(img, 0.5, segmentation_image, 0.7, 0)

        out_path = os.path.join(args.out_dir, filename)
        if args.save: 
            cv2.imwrite(out_path, combined)

        # Save JSON metadata and segmentation map
        json_data = {
            "filename": filename,
            "class": image_class,
            "shape": img.shape[:2],
            "model": f"sapiens-{args.model}",
            "dtype": args.dtype,
            "classes": segmentation_classes,
            "polygons": polygons
        }
        json_path = os.path.join(args.out_dir, os.path.splitext(filename)[0] + ".json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        print(f"Saved results to {args.out_dir}")

        # Show only if processing a single image
        if len(images_to_process) == 1:
            cv2.namedWindow("Segmentation Map", cv2.WINDOW_NORMAL)
            cv2.imshow("Segmentation Map", combined)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()