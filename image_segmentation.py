import torch
import cv2
import os
import argparse
from glob import glob
from imread_from_url import imread_from_url

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map

def main():
    parser = argparse.ArgumentParser(description="Sapiens Image Segmentation")
    parser.add_argument("--url", type=str, help="URL of the image to segment")
    parser.add_argument("--img_dir", type=str, help="Directory containing images to segment")
    parser.add_argument("--out_dir", type=str, default="./runs", help="Output directory for segmented images")
    parser.add_argument("--model", type=str, default="1b", choices=["03b", "06b", "1b"], help="Model size: 03b, 06b, or 1b")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type for inference")
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

        print(f"Processing {path}...")
        segmentation_map = estimator(img)

        segmentation_image = draw_segmentation_map(segmentation_map)
        combined = cv2.addWeighted(img, 0.5, segmentation_image, 0.7, 0)

        out_path = os.path.join(args.out_dir, f"seg_{filename}")
        cv2.imwrite(out_path, combined)
        print(f"Saved to {out_path}")

        # Show only if processing a single image
        if len(images_to_process) == 1:
            cv2.namedWindow("Segmentation Map", cv2.WINDOW_NORMAL)
            cv2.imshow("Segmentation Map", combined)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()