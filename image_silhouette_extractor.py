import torch
import cv2
import os
import argparse
import json
import math
import numpy as np
from tqdm import tqdm
from glob import glob
from imread_from_url import imread_from_url

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType
from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType

# COCO-style keypoint mapping (assuming first 17 keypoints follow standard layout)
KEYPOINTS_MAP = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Define the limbs using keypoint pairs
LIMBS = {
    "upperarm_left": ("left_shoulder", "left_elbow"),
    "lowerarm_left": ("left_elbow", "left_wrist"),
    "upperarm_right": ("right_shoulder", "right_elbow"),
    "lowerarm_right": ("right_elbow", "right_wrist"),
    "upperleg_left": ("left_hip", "left_knee"),
    "lowerleg_left": ("left_knee", "left_ankle"),
    "upperleg_right": ("right_hip", "right_knee"),
    "lowerleg_right": ("right_knee", "right_ankle"),
    "torso_left": ("left_shoulder", "left_hip"),
    "torso_right": ("right_shoulder", "right_hip"),
}

def find_edge(mask, start_x, start_y, dx, dy, step=1, max_dist=2000):
    """ Cast a ray to find the intersection with the background (mask == 0). """
    x, y = float(start_x), float(start_y)
    h, w = mask.shape
    
    for _ in range(max_dist):
        curr_x, curr_y = int(x), int(y)
        
        # Check image boundaries
        if curr_x < 0: curr_x = 0
        if curr_x >= w: curr_x = w - 1
        if curr_y < 0: curr_y = 0
        if curr_y >= h: curr_y = h - 1

        # Check if we hit background (0) or boundary
        if mask[curr_y, curr_x] == 0 or \
           curr_x == 0 or curr_x == w - 1 or curr_y == 0 or curr_y == h - 1:
            return [curr_x, curr_y]

        x += dx * step
        y += dy * step

    return [int(x), int(y)]

def main():
    parser = argparse.ArgumentParser(description="Sapiens Image Silhouette Extractor")
    parser.add_argument("--url", type=str, help="URL of the image to process")
    parser.add_argument("--img_dir", type=str, help="Directory containing images to process")
    parser.add_argument("--out_dir", type=str, default="./runs", help="Output directory for processed images/json")
    parser.add_argument("--model", type=str, default="1b", choices=["03b", "06b", "1b"], help="Model size: 03b, 06b, or 1b")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type for inference")
    parser.add_argument("--save", action="store_true", default=True, help="Save the output image and json (default: True)")
    parser.add_argument("--no-save", action="store_false", dest="save", help="Do not save the output")

    args = parser.parse_args()

    # Map model size for both Segmentation and Pose
    seg_model_map = {
        "03b": SapiensSegmentationType.SEGMENTATION_03B,
        "06b": SapiensSegmentationType.SEGMENTATION_06B,
        "1b": SapiensSegmentationType.SEGMENTATION_1B,
    }
    
    pose_model_map = {
        "03b": SapiensPoseEstimationType.POSE_ESTIMATION_03B,
        "06b": SapiensPoseEstimationType.POSE_ESTIMATION_06B,
        "1b": SapiensPoseEstimationType.POSE_ESTIMATION_1B,
    }

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    dtype = dtype_map[args.dtype]
    
    print("Loading models...")
    seg_estimator = SapiensSegmentation(seg_model_map[args.model], dtype=dtype)
    pose_estimator = SapiensPoseEstimation(pose_model_map[args.model], dtype=dtype)
    os.makedirs(args.out_dir, exist_ok=True)

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

        # 1. Get bounding boxes and keypoints
        bboxes = pose_estimator.detector.detect(img)
        pose_result_img, keypoints = pose_estimator(img)
        
        if bboxes is None or len(bboxes) == 0:
            print(f"Warning: No person detected for {filename}!")
            continue
            
        bbox = bboxes[0] # Assuming single person for silhouette

        # 2. Isolate the human using segmentation
        segmentation_map = seg_estimator(img)
        human_mask = (segmentation_map > 0).astype(np.uint8) # 0 is background
        
        # Ensure mask is perfectly aligned with original image shape
        ih, iw = img.shape[:2]
        if human_mask.shape[:2] != (ih, iw):
            human_mask = cv2.resize(human_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

        # Sapiens returns a list containing a dict of {keypoint_name: (x, y, score)}
        if isinstance(keypoints, (list, tuple)) and len(keypoints) > 0:
            kpts = keypoints[0] # Take first person
        else:
            kpts = keypoints
            
        if not kpts:
            print(f"Warning: No keypoints found for {filename}!")
            continue

        # Draw on pose_result_img so we can visually verify alignment with the original stick-figure!
        draw_img = pose_result_img.copy() if pose_result_img is not None else img.copy()
        limb_silhouettes = {}

        # Sapiens/MMPose exact crop padding logic reverse-engineered
        bx1, by1, bx2, by2 = map(float, bbox[:4])
        bw = bx2 - bx1
        bh = by2 - by1
        px1 = bx1
        py1 = by1

        # 3 & 4. Slicing limbs and calculating intersections
        for limb_name, (kp1_name, kp2_name) in LIMBS.items():
            
            # Fetch coordinates based on dict string keys or array integer indices
            if isinstance(kpts, dict) and kp1_name in kpts and kp2_name in kpts:
                x1, y1 = kpts[kp1_name][:2]
                x2, y2 = kpts[kp2_name][:2]
            else:
                idx1, idx2 = KEYPOINTS_MAP.get(kp1_name), KEYPOINTS_MAP.get(kp2_name)
                if hasattr(kpts, "__len__") and idx1 is not None and idx2 is not None and idx1 < len(kpts) and idx2 < len(kpts):
                    x1, y1 = kpts[idx1][:2]
                    x2, y2 = kpts[idx2][:2]
                else:
                    continue

            # Scale keypoints from the 192x256 heatmap space back to original padded image space
            x1 = (x1 / 192.0) * bw + px1
            y1 = (y1 / 256.0) * bh + py1
            x2 = (x2 / 192.0) * bw + px1
            y2 = (y2 / 256.0) * bh + py1

            # Vector of the limb
            vx, vy = x2 - x1, y2 - y1
            mag = math.hypot(vx, vy)
            
            if mag == 0:
                continue

            # Normalized perpendicular vector
            px, py = -vy / mag, vx / mag
            
            limb_points = []
            
            # Slice in steps of 10% (0.0 to 1.0)
            for step in range(11):
                t = step / 10.0
                cx_pt = x1 + t * vx
                cy_pt = y1 + t * vy
                
                # Find intersection on both sides (+perpendicular, -perpendicular)
                pt_left = find_edge(human_mask, cx_pt, cy_pt, px, py)
                pt_right = find_edge(human_mask, cx_pt, cy_pt, -px, -py)
                
                limb_points.append([pt_left, pt_right])

                # Draw on image
                cv2.circle(draw_img, (pt_left[0], pt_left[1]), 3, (0, 0, 255), -1)  # Red for left edge
                cv2.circle(draw_img, (pt_right[0], pt_right[1]), 3, (0, 255, 0), -1) # Green for right edge
                cv2.line(draw_img, (pt_left[0], pt_left[1]), (pt_right[0], pt_right[1]), (255, 0, 0), 1) # Blue line

            limb_silhouettes[limb_name] = limb_points

        # 5. Save results
        if args.save:
            # Save the image with drawn silhouette slice-points
            out_img_path = os.path.join(args.out_dir, f"silhouettes_{filename}")
            cv2.imwrite(out_img_path, draw_img)

            # Save JSON
            json_path = os.path.join(args.out_dir, os.path.splitext(filename)[0] + "_silhouettes.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(limb_silhouettes, f, indent=4)

        if len(images_to_process) == 1:
            cv2.namedWindow("Silhouette Extractor", cv2.WINDOW_NORMAL)
            cv2.imshow("Silhouette Extractor", draw_img)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()