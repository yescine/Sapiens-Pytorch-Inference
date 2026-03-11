import torch
import cv2
import os
import argparse
from tqdm import tqdm
from glob import glob
from imread_from_url import imread_from_url

from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType

def main():
    parser = argparse.ArgumentParser(description="Sapiens Image Pose Estimation")
    parser.add_argument("--url", type=str, help="URL of the image to process")
    parser.add_argument("--img_dir", type=str, help="Directory containing images to process")
    parser.add_argument("--out_dir", type=str, default="./runs", help="Output directory for processed images")
    parser.add_argument("--model", type=str, default="1b", choices=["03b", "06b", "1b"], help="Model size: 03b, 06b, or 1b")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type for inference")
    parser.add_argument("--save", action="store_true", default=True, help="Save the output image (default: True)")
    parser.add_argument("--no-save", action="store_false", dest="save", help="Do not save the output image")

    args = parser.parse_args()

    # Map model size
    model_map = {
        "03b": SapiensPoseEstimationType.POSE_ESTIMATION_03B,
        "06b": SapiensPoseEstimationType.POSE_ESTIMATION_06B,
        "1b": SapiensPoseEstimationType.POSE_ESTIMATION_1B,
    }

    # Map dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    dtype = dtype_map[args.dtype]
    model_type = model_map[args.model]

    estimator = SapiensPoseEstimation(model_type, dtype=dtype)
    os.makedirs(args.out_dir, exist_ok=True)

    images_to_process = []
    if args.url:
        images_to_process.append(("url", args.url))

    if args.img_dir:
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        for ext in extensions:
            images_to_process.extend([("file", f) for f in glob(os.path.join(args.img_dir, ext))])

    if not args.url and not args.img_dir:
        images_to_process.append(("url", "https://learnopencv.com/wp-content/uploads/2024/09/football-soccer-scaled.jpg"))

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

        result_img, keypoints = estimator(img)

        out_path = os.path.join(args.out_dir, filename)
        if args.save:
            cv2.imwrite(out_path, result_img)

        if len(images_to_process) == 1:
            cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
            cv2.imshow("Pose Estimation", result_img)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()
