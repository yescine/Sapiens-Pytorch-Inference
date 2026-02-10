import time
from enum import Enum

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import create_preprocessor, TaskType, download_hf_model


class SapiensSegmentationType(Enum):
    SEGMENTATION_03B = "sapiens-seg-0.3b-torchscript/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"
    SEGMENTATION_06B = "sapiens-seg-0.6b-torchscript/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
    SEGMENTATION_1B = "sapiens-seg-1b-torchscript/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"


random = np.random.RandomState(11)

classes = [
    "background",        # 0
    "apparel",          # 1
    "face_neck",        # 2
    "hair",             # 3

    "l_foot",           # 4
    "l_hand",           # 5
    "l_lower_arm",      # 6
    "l_lower_leg",      # 7
    "l_shoe",           # 8
    "l_sock",           # 9
    "l_upper_arm",      # 10
    "l_upper_leg",      # 11

    "clothing_lower",   # 12

    "r_foot",           # 13
    "r_hand",           # 14
    "r_lower_arm",      # 15
    "r_lower_leg",      # 16
    "r_shoe",           # 17
    "r_sock",           # 18
    "r_upper_arm",      # 19
    "r_upper_leg",      # 20

    "torso",            # 21
    "clothing_upper",   # 22

    "lower_lip",        # 23
    "upper_lip",        # 24
    "lower_teeth",      # 25
    "upper_teeth",      # 26
    "tongue"           # 27
]

colors = random.randint(0, 255, (len(classes) - 1, 3))
colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
colors = colors[:, ::-1]


def draw_segmentation_map(segmentation_map: np.ndarray) -> np.ndarray:
    h, w = segmentation_map.shape
    segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        segmentation_img[segmentation_map == i] = color

    return segmentation_img

def segmentation_to_polygons(segmentation_map: np.ndarray,
                             min_area: int = 10,
                             epsilon_ratio: float = 0.001):
    """
    Convert segmentation map into polygons per class.

    epsilon_ratio controls simplification:
        smaller = more accurate
        larger = fewer points
    """

    polygons = {}
    
    segmentation_map = segmentation_map.astype(np.uint8)

    for class_id in np.unique(segmentation_map):

        # if class_id == 0:
        #     continue  # skip background

        mask = (segmentation_map == class_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        class_polys = []

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < min_area:
                continue

            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            polygon = approx.reshape(-1, 2).tolist()

            if len(polygon) >= 3:
                class_polys.append(polygon)

        if class_polys:
            polygons[classes[class_id]] = class_polys

    return polygons

def polygons_to_mask(image_shape, polygons_dict):
    """
    Reconstruct segmentation mask from polygons.

    image_shape: (H, W)
    polygons_dict: {
        "Torso": [ [[x,y], [x,y]...], ... ],
        ...
    }
    """

    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for class_name, polys in polygons_dict.items():

        class_id = classes.index(class_name)

        for poly in polys:

            pts = np.array(poly, dtype=np.int32)

            if pts.shape[0] < 3:
                continue

            cv2.fillPoly(mask, [pts], class_id)

    return mask


def postprocess_segmentation(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    result = results[0].cpu()

    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    # Perform argmax to get the segmentation map
    segmentation_map = logits.argmax(dim=0, keepdim=True)

    # Covert to numpy array
    # segmentation_map = segmentation_map.to(torch.int16).cpu().numpy().squeeze()
    segmentation_map = segmentation_map.cpu().numpy().astype(np.uint8).squeeze()

    return segmentation_map


class SapiensSegmentation():
    def __init__(self,
                 type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_1B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        path = download_hf_model(type.value)
        model = torch.jit.load(path)
        model = model.eval()
        self.model = model.to(device).to(dtype)
        self.device = device
        self.dtype = dtype
        self.preprocessor = create_preprocessor(input_size=(1024, 768))  # Only these values seem to work well

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():
            results = self.model(tensor)
        segmentation_map = postprocess_segmentation(results, img.shape[:2])

        print(f"Segmentation inference took: {time.perf_counter() - start:.4f} seconds")
        return segmentation_map


if __name__ == "__main__":
    type = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "test.jpg"
    img = cv2.imread(img_path)

    model_type = SapiensSegmentationType.SEGMENTATION_1B
    estimator = SapiensSegmentation(model_type)

    start = time.perf_counter()
    segmentations = estimator(img)
    print(f"Time taken: {time.perf_counter() - start:.4f} seconds")

    segmentation_img = draw_segmentation_map(segmentations)
    combined = cv2.addWeighted(img, 0.5, segmentation_img, 0.5, 0)

    cv2.imshow("segmentation_map", combined)
    cv2.waitKey(0)
