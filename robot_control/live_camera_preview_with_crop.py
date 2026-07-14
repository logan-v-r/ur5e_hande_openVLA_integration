from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

# Start with 0, then test 1, 2, and 3 until you find the normal full-color
# RealSense RGB view.
CAMERA_DEVICE = 4

# OpenVLA's common square image size. This preview shows the spatial crop and
# resize only; model normalization is not shown because it is not useful for
# human visual inspection.
OPENVLA_IMAGE_SIZE = 224

# Optional: save still images here when you press S.
OUTPUT_DIR = Path.home() / "workspaces" / "openvla" / "ur5_rtde" / "logs"


# ---------------------------------------------------------------------------
# Image-preview helpers
# ---------------------------------------------------------------------------

def center_square_crop(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return the centered square region that will be retained before resizing.

    The returned crop bounds are ``(left, top, right, bottom)`` in the original
    camera frame. This makes it easy to draw the retained region on the native
    preview for visual review.
    """
    height, width = frame.shape[:2]
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return frame[top:bottom, left:right], (left, top, right, bottom)


def make_openvla_preview(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Create a 224x224 spatial preview using center-crop then resize.

    This matches the geometry of a standard square center-crop preprocessing
    path. It is intended to reveal whether important workspace content is cut
    off before fine-tuning; it does not apply model normalization.
    """
    square_crop, crop_bounds = center_square_crop(frame)
    preview = cv2.resize(
        square_crop,
        (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return preview, crop_bounds


def draw_crop_boundary(frame: np.ndarray, crop_bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Draw the center-crop boundary on a copy of the native camera frame."""
    left, top, right, bottom = crop_bounds
    annotated = frame.copy()
    cv2.rectangle(annotated, (left, top), (right - 1, bottom - 1), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        "OpenVLA 224x224 retained area",
        (left + 8, max(top + 25, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def build_side_by_side_view(native_frame: np.ndarray, openvla_frame: np.ndarray) -> np.ndarray:
    """Place the crop-marked native feed beside an enlarged 224x224 preview."""
    native_height = native_frame.shape[0]
    preview_large = cv2.resize(
        openvla_frame,
        (native_height, native_height),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.putText(
        preview_large,
        "OpenVLA spatial preview: 224 x 224",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.hconcat([native_frame, preview_large])


# ---------------------------------------------------------------------------
# Main preview loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Show the native camera view and the corresponding 224x224 crop preview.

    Controls:
      Q or Esc  -> close the preview
      S         -> save both the native frame and the 224x224 preview
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not capture.isOpened():
        sys.exit(f"Could not open camera device: {CAMERA_DEVICE}")

    print(f"Showing live preview from camera device: {CAMERA_DEVICE}")
    print("Left: native frame with green retained-area boundary.")
    print("Right: 224 x 224 center-crop preview for OpenVLA framing review.")
    print("Press Q or Esc to quit. Press S to save both views.")

    image_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Could not read a camera frame.")
                break

            openvla_frame, crop_bounds = make_openvla_preview(frame)
            native_annotated = draw_crop_boundary(frame, crop_bounds)
            display = build_side_by_side_view(native_annotated, openvla_frame)

            cv2.imshow("Native Camera + OpenVLA 224x224 Preview", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                native_path = OUTPUT_DIR / f"camera_preview_native_{image_number:03d}.jpg"
                openvla_path = OUTPUT_DIR / f"camera_preview_openvla_224_{image_number:03d}.jpg"

                native_ok = cv2.imwrite(str(native_path), frame)
                openvla_ok = cv2.imwrite(str(openvla_path), openvla_frame)

                if native_ok and openvla_ok:
                    print(f"Saved native:  {native_path}")
                    print(f"Saved 224 view: {openvla_path}")
                    image_number += 1
                else:
                    print("Could not save one or both preview images.")

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
