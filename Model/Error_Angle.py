#!/usr/bin/env python3
"""
YOLO OBB angle measurement using a live camera.

Flow:
  1. Open camera only. This is IDLE mode.
  2. Press A and enter the ground-truth angle.
  3. Press M to enable the YOLO OBB model. This is MODEL mode.
  4. Press SPACE in MODEL mode to sample several frames.
  5. Print the circular average angle and error.
  6. Return to IDLE mode and enter the next angle.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import PySpin

    _PYSPIN_AVAILABLE = True
except ImportError:
    _PYSPIN_AVAILABLE = False


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")

CONF_THRES = 0.7
N_SAMPLES = 15
ANGLE_PERIOD = 180.0
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CLASS_NAMES = {
    0: "Blade",
    1: "bottle",
    2: "scalpel10",
    3: "scalpel11",
    4: "scalpel15",
}

CLASS_COLORS = {
    0: (200, 80, 255),
    1: (0, 255, 0),
    2: (255, 100, 0),
    3: (0, 100, 255),
    4: (255, 0, 255),
}


def make_display_frame(frame: np.ndarray) -> np.ndarray:
    """Fit camera frame into a fixed-size canvas without changing aspect ratio."""
    src_h, src_w = frame.shape[:2]
    scale = min(DISPLAY_WIDTH / src_w, DISPLAY_HEIGHT / src_h)
    resized_w = int(src_w * scale)
    resized_h = int(src_h * scale)

    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    x0 = (DISPLAY_WIDTH - resized_w) // 2
    y0 = (DISPLAY_HEIGHT - resized_h) // 2
    canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return canvas


def draw_reference_line(frame: np.ndarray) -> np.ndarray:
    """Draw center-based angle guide lines for manual angle measurement."""
    out = frame.copy()
    height, width = out.shape[:2]
    cx, cy = width // 2, height // 2
    line_length = int(max(width, height) * 1.5)

    for angle_deg in range(0, 180, 30):
        theta = math.radians(angle_deg)
        dx = int(math.cos(theta) * line_length)
        dy = int(math.sin(theta) * line_length)
        pt1 = (cx - dx, cy + dy)
        pt2 = (cx + dx, cy - dy)
        color = (0, 255, 255) if angle_deg in (0, 90) else (0, 180, 255)
        thickness = 2 if angle_deg in (0, 90) else 1

        cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

        label_x = int(cx + math.cos(theta) * min(width, height) * 0.35)
        label_y = int(cy - math.sin(theta) * min(width, height) * 0.35)
        label_x = max(5, min(width - 45, label_x))
        label_y = max(18, min(height - 8, label_y))
        cv2.putText(
            out,
            f"{angle_deg}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
    return out


@dataclass
class Detection:
    cls: int
    name: str
    cx: float
    cy: float
    width: float
    height: float
    raw_angle_deg: float
    angle_deg: float
    conf: float


class SpinnakerCamera:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        if cam_list.GetSize() == 0:
            raise RuntimeError("Spinnaker camera not found")

        self.cam = cam_list[0]
        self.cam.Init()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()
        cam_list.Clear()

    def read(self):
        img_result = self.cam.GetNextImage(1000)
        if img_result.IsIncomplete():
            img_result.Release()
            return None

        frame = img_result.GetNDArray()
        img_result.Release()
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def release(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.system.ReleaseInstance()


class OpenCVCamera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV camera {index} could not be opened")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


def open_camera():
    """Open Spinnaker first, then fall back to a normal OpenCV camera."""
    if _PYSPIN_AVAILABLE:
        try:
            cam = SpinnakerCamera()
            print("[camera] Spinnaker camera")
            return cam
        except Exception as exc:
            print(f"[camera] Spinnaker failed: {exc}. Falling back to OpenCV.")

    cam = OpenCVCamera(0)
    print("[camera] OpenCV camera 0")
    return cam


def load_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[error] Model file not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    print(f"[model] loaded: {MODEL_PATH}")
    return model


def normalize_angle_180(angle_deg: float) -> float:
    return angle_deg % ANGLE_PERIOD


def long_axis_angle_deg(width: float, height: float, raw_angle_rad: float) -> float:
    """
    Convert YOLO OBB xywhr angle to a long-axis object angle.

    Ultralytics xywhr gives the rectangle rotation. For long tools, the
    direction we usually care about is the long side. If h > w, the long side
    is perpendicular to the reported width side, so add 90 degrees.
    """
    angle = math.degrees(raw_angle_rad)
    if height > width:
        angle += 90.0
    return normalize_angle_180(angle)


def circular_mean_deg(angles_deg: Iterable[float], period: float = ANGLE_PERIOD) -> float:
    angles = np.asarray(list(angles_deg), dtype=float)
    theta = angles * (2 * np.pi / period)
    mean_sin = np.mean(np.sin(theta))
    mean_cos = np.mean(np.cos(theta))
    mean_theta = math.atan2(mean_sin, mean_cos)
    return (mean_theta * period / (2 * np.pi)) % period


def circular_diff_signed(a: float, b: float, period: float = ANGLE_PERIOD) -> float:
    """Return the signed minimum difference a - b in [-period/2, period/2)."""
    return (a - b + period / 2) % period - period / 2


def circular_diff_abs(a: float, b: float, period: float = ANGLE_PERIOD) -> float:
    return abs(circular_diff_signed(a, b, period))


def parse_detections(results) -> list[Detection]:
    detections = []
    for result in results:
        if result.obb is None:
            continue

        xywhr = result.obb.xywhr.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        clss = result.obb.cls.cpu().numpy().astype(int)

        for (cx, cy, width, height, raw_angle), conf, cls in zip(xywhr, confs, clss):
            cls = int(cls)
            name = CLASS_NAMES.get(cls, str(cls))
            detections.append(
                Detection(
                    cls=cls,
                    name=name,
                    cx=float(cx),
                    cy=float(cy),
                    width=float(width),
                    height=float(height),
                    raw_angle_deg=normalize_angle_180(math.degrees(raw_angle)),
                    angle_deg=long_axis_angle_deg(width, height, raw_angle),
                    conf=float(conf),
                )
            )
    return detections


def best_detection(results, target_cls_name=None) -> Detection | None:
    """Pick the highest-confidence detection to measure."""
    best = None
    for det in parse_detections(results):
        if target_cls_name is not None:
            if det.name != target_cls_name:
                continue
        elif det.name == "Blade":
            continue

        if best is None or det.conf > best.conf:
            best = det
    return best


def draw_results(frame: np.ndarray, results, target_cls_name=None) -> np.ndarray:
    out = draw_reference_line(frame)
    best = best_detection(results, target_cls_name)
    best_key = None
    if best is not None:
        best_key = (best.cls, round(best.cx, 1), round(best.cy, 1), round(best.conf, 4))

    for result in results:
        if result.obb is None:
            continue

        corners_all = result.obb.xyxyxyxy.cpu().numpy().astype(np.int32)
        detections = parse_detections([result])

        for corners, det in zip(corners_all, detections):
            color = CLASS_COLORS.get(det.cls, (220, 220, 220))
            det_key = (det.cls, round(det.cx, 1), round(det.cy, 1), round(det.conf, 4))
            thickness = 3 if det_key == best_key else 2
            label = f"{det.name} {det.angle_deg:.1f}deg conf={det.conf:.2f}"

            cv2.polylines(out, [corners], isClosed=True, color=color, thickness=thickness)
            cv2.circle(out, (int(det.cx), int(det.cy)), 5, color, -1)
            cv2.putText(
                out,
                label,
                (int(det.cx) - 70, int(det.cy) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return out


def take_measurement(cam, model, n_samples, conf_thres, target_cls_name=None):
    angles = []
    confs = []
    win_name = "Error_Angle"

    for i in range(n_samples):
        frame = cam.read()
        if frame is None:
            continue
        frame = make_display_frame(frame)

        results = model(frame, conf=conf_thres, verbose=False)
        det = best_detection(results, target_cls_name)

        disp = draw_results(frame, results, target_cls_name)
        cv2.putText(
            disp,
            f"measuring {i + 1}/{n_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if det is not None:
            angles.append(det.angle_deg)
            confs.append(det.conf)

        cv2.putText(
            disp,
            "MODEL | sampling angles",
            (10, disp.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(win_name, disp)
        cv2.waitKey(1)

    return angles, confs


def parse_args():
    parser = argparse.ArgumentParser(description="Measure YOLO OBB angle from a live camera.")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Frames sampled for one measurement.")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help="YOLO confidence threshold.")
    parser.add_argument(
        "--class",
        dest="cls_name",
        default=None,
        choices=list(CLASS_NAMES.values()),
        help="Target class. If omitted, the best non-Blade detection is used.",
    )
    return parser.parse_args()


def read_target_angle():
    while True:
        raw = input("Ground-truth angle in degrees (0~180, blank to cancel): ").strip()
        if raw == "":
            print("[angle] canceled")
            return None

        try:
            angle = normalize_angle_180(float(raw))
        except ValueError:
            print("[angle] Please enter a number.")
            continue

        print(f"[angle] target={angle:.1f} deg")
        return angle


def main():
    args = parse_args()
    cam = open_camera()
    model = None
    mode = "IDLE"
    target_angle = None
    measurement_count = 0

    print("\n[guide]")
    print("  IDLE  : camera preview only")
    print("  A     : enter ground-truth angle")
    print("  M     : enable model and enter MODEL mode after angle input")
    print("  SPACE : sample angles in MODEL mode, print average, then return to IDLE")
    print("  Q/ESC : quit")
    print("  Angle is measured from the OBB long axis, modulo 180 degrees.\n")

    try:
        cv2.namedWindow("Error_Angle", cv2.WINDOW_AUTOSIZE)
        while True:
            frame = cam.read()
            if frame is None:
                continue
            frame = make_display_frame(frame)

            if mode == "MODEL":
                results = model(frame, conf=args.conf, verbose=False)
                det = best_detection(results, args.cls_name)
                disp = draw_results(frame, results, args.cls_name)

                if det is not None:
                    status = f"target={det.name} angle={det.angle_deg:.1f}deg conf={det.conf:.2f}"
                else:
                    status = "target=none"

                cv2.putText(
                    disp,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                footer = (
                    f"MODEL | GT={target_angle:.1f}deg | measurements={measurement_count} | "
                    f"SPACE: average | I: idle | Q/ESC: quit"
                )
            else:
                disp = draw_reference_line(frame)
                if target_angle is None:
                    footer = f"IDLE | measurements={measurement_count} | A: enter angle | Q/ESC: quit"
                else:
                    footer = (
                        f"IDLE | GT={target_angle:.1f}deg | measurements={measurement_count} | "
                        f"M: model on | A: change angle | Q/ESC: quit"
                    )

            cv2.putText(
                disp,
                footer,
                (10, disp.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Error_Angle", disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("a"), ord("A")):
                mode = "IDLE"
                target_angle = read_target_angle()

            elif key in (ord("m"), ord("M")):
                if target_angle is None:
                    print("[idle] Press A and enter the ground-truth angle first.")
                    continue

                if model is None:
                    model = load_model()
                mode = "MODEL"
                print(f"[state] MODEL on | GT={target_angle:.1f} deg")

            elif key in (ord("i"), ord("I")):
                mode = "IDLE"
                print("[state] IDLE")

            elif key == 32:
                if mode != "MODEL":
                    print("[idle] Press M before measuring.")
                    continue

                angles, confs = take_measurement(cam, model, args.samples, args.conf, args.cls_name)
                if not angles:
                    print("[fail] No detection. Check camera view, lighting, class, or confidence threshold.")
                    mode = "IDLE"
                    print("[state] IDLE")
                    continue

                avg_angle = circular_mean_deg(angles)
                signed_error = circular_diff_signed(avg_angle, target_angle)
                error = abs(signed_error)
                mean_conf = float(np.mean(confs))
                measurement_count += 1
                print(
                    f"[result #{measurement_count}] "
                    f"GT={target_angle:.1f} deg | "
                    f"avg_angle={avg_angle:.2f} deg | "
                    f"error={error:.2f} deg | "
                    f"signed_error={signed_error:+.2f} deg | "
                    f"detected={len(angles)}/{args.samples} | "
                    f"mean_conf={mean_conf:.2f}"
                )
                mode = "IDLE"
                target_angle = None
                print("[state] IDLE")

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
