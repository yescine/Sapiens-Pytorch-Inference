# cvat_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any
import xml.etree.ElementTree as ET


Point = Union[Tuple[int, int], Tuple[float, float], List[Union[int, float]]]


def create_cvat_root(version: str = "1.1") -> ET.Element:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = version
    return root


def create_cvat_image(
    root: ET.Element,
    image_id: int,
    filename: str,
    width: int,
    height: int,
) -> ET.Element:
    return ET.SubElement(
        root,
        "image",
        attrib={
            "id": str(image_id),
            "name": filename,
            "width": str(int(width)),
            "height": str(int(height)),
        },
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_polygon_points(
    pts: Sequence[Point],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    clamp: bool = True,
) -> List[Tuple[float, float]]:
    """
    Normalize various point shapes to [(x,y), ...] as floats.
    Optionally clamps to image bounds if width/height provided.
    """
    out: List[Tuple[float, float]] = []

    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x = float(p[0])
            y = float(p[1])
        else:
            continue

        if clamp and width is not None and height is not None:
            x = _clamp(x, 0.0, float(width - 1))
            y = _clamp(y, 0.0, float(height - 1))

        out.append((x, y))

    return out


def add_cvat_polygon(
    image_el: ET.Element,
    label: str,
    polygon_pts: Sequence[Point],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    clamp: bool = True,
    occluded: int = 0,
    source: str = "manual",
) -> bool:
    """
    Adds a CVAT <polygon> if it has >= 3 valid points.
    Returns True if added, False otherwise.
    """
    pts = normalize_polygon_points(
        polygon_pts, width=width, height=height, clamp=clamp
    )
    if len(pts) < 3:
        return False

    # CVAT accepts float coords. Keep 2 decimals to reduce filesize.
    pts_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)

    ET.SubElement(
        image_el,
        "polygon",
        attrib={
            "label": str(label),
            "points": pts_str,
            "occluded": str(int(occluded)),
            "source": str(source),
        },
    )
    return True


def write_cvat_xml(root: ET.Element, out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(
        out_path, encoding="utf-8", xml_declaration=True
    )
    return out_path