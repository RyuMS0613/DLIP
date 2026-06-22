#!/usr/bin/env python3
"""
Camera-only calibration validation.

Use this after calibration.py has created homography.npy.

Flow:
  1. Open camera.
  2. Click a known point in the image.
  3. The script converts pixel (u, v) to predicted robot XY using homography.npy.
  4. Enter the real robot XY for that clicked point.
  5. The script prints and stores the error.
  6. Repeat for multiple points, then print/save overall error statistics.

Keys:
  Left click  Add one validation point
  P           Print all records and summary
  S           Save CSV
  Z           Undo last point
  R           Reset all points
  Q / ESC     Quit
"""

import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

try:
    import PySpin

    _PYSPIN_AVAILABLE = True
except ImportError:
    _PYSPIN_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOMOGRAPHY_NPY = os.path.join(BASE_DIR, "homography.npy")
RESULT_DIR = os.path.join(BASE_DIR, "results")
PANEL_W = 280


class SpinnakerCamera:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        if cam_list.GetSize() == 0:
            self.system.ReleaseInstance()
            raise RuntimeError("No Spinnaker camera found")

        self.cam = cam_list.GetByIndex(0)
        self.cam.Init()
        self.cam.BeginAcquisition()
        self.processor = PySpin.ImageProcessor()
        self.processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    def read(self) -> Optional[np.ndarray]:
        img = self.cam.GetNextImage(1000)
        if img.IsIncomplete():
            img.Release()
            return None

        frame = self.processor.Convert(img, PySpin.PixelFormat_BGR8).GetNDArray().copy()
        img.Release()
        return frame

    def release(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()


class OpenCVCamera:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


def open_camera():
    if _PYSPIN_AVAILABLE:
        try:
            cam = SpinnakerCamera()
            print("[CAM] Spinnaker camera OK")
            return cam
        except Exception as exc:
            print(f"[CAM] Spinnaker failed ({exc}), falling back to OpenCV")

    cam = OpenCVCamera()
    print("[CAM] OpenCV VideoCapture OK")
    return cam


def load_homography() -> np.ndarray:
    if not os.path.exists(HOMOGRAPHY_NPY):
        raise SystemExit(f"[ERROR] homography not found: {HOMOGRAPHY_NPY}")

    H = np.load(HOMOGRAPHY_NPY)
    print(f"[H] loaded: {HOMOGRAPHY_NPY}")
    return H


def pixel_to_robot(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    src = np.float32([[[u, v]]])
    dst = cv2.perspectiveTransform(src, H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


@dataclass
class ValidationRecord:
    pixel_u: float
    pixel_v: float
    pred_x: float
    pred_y: float
    actual_x: float
    actual_y: float

    @property
    def dx(self) -> float:
        return self.pred_x - self.actual_x

    @property
    def dy(self) -> float:
        return self.pred_y - self.actual_y

    @property
    def error(self) -> float:
        return float(np.hypot(self.dx, self.dy))


@dataclass
class ValidationStore:
    records: list[ValidationRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.records)

    def add(self, record: ValidationRecord):
        self.records.append(record)

    def undo(self):
        if self.records:
            removed = self.records.pop()
            print(f"[UNDO] removed point #{self.n + 1}, error={removed.error:.2f} mm")

    def reset(self):
        self.records.clear()

    def errors(self) -> np.ndarray:
        return np.array([r.error for r in self.records], dtype=float)

    def stats(self) -> dict:
        errors = self.errors()
        if errors.size == 0:
            return {}
        return {
            "mean": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "max": float(np.max(errors)),
            "std": float(np.std(errors)),
        }


clicks: list[tuple[int, int]] = []
camera_width = 0


def on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if camera_width == 0 or x < camera_width:
            clicks.append((x, y))


def ask_actual_xy(pixel: tuple[float, float], pred: tuple[float, float]) -> Optional[tuple[float, float]]:
    print("\n" + "-" * 70)
    print(f"Clicked pixel       : ({pixel[0]:.0f}, {pixel[1]:.0f})")
    print(f"Predicted robot XY : ({pred[0]:.2f}, {pred[1]:.2f}) mm")
    raw = input("Actual robot XY [mm], e.g. '250 50' (Enter = skip): ").strip()
    if not raw:
        print("[SKIP] point skipped")
        return None

    try:
        parts = raw.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError
        return float(parts[0]), float(parts[1])
    except ValueError:
        print("[SKIP] invalid input")
        return None


def print_record(record: ValidationRecord, idx: int):
    print(
        f"[V{idx}] pixel=({record.pixel_u:.0f}, {record.pixel_v:.0f}) | "
        f"pred=({record.pred_x:.2f}, {record.pred_y:.2f}) mm | "
        f"actual=({record.actual_x:.2f}, {record.actual_y:.2f}) mm | "
        f"dx={record.dx:+.2f}, dy={record.dy:+.2f}, error={record.error:.2f} mm"
    )


def print_summary(store: ValidationStore):
    print("\n" + "=" * 92)
    print("Calibration Validation Summary")
    print("=" * 92)
    print(
        f"{'#':<4}{'pixel(u,v)':<18}{'pred XY(mm)':<24}"
        f"{'actual XY(mm)':<24}{'dx':>8}{'dy':>8}{'err':>8}"
    )
    print("-" * 92)

    for i, r in enumerate(store.records, 1):
        print(
            f"{i:<4}"
            f"({r.pixel_u:>5.0f},{r.pixel_v:>5.0f})    "
            f"({r.pred_x:>8.2f},{r.pred_y:>8.2f})    "
            f"({r.actual_x:>8.2f},{r.actual_y:>8.2f})    "
            f"{r.dx:>8.2f}{r.dy:>8.2f}{r.error:>8.2f}"
        )

    stats = store.stats()
    if stats:
        print("-" * 92)
        print(f"Mean error : {stats['mean']:.2f} mm")
        print(f"RMSE       : {stats['rmse']:.2f} mm")
        print(f"Max error  : {stats['max']:.2f} mm")
        print(f"Std dev    : {stats['std']:.2f} mm")
    else:
        print("(no validation points)")
    print("=" * 92 + "\n")


def save_csv(store: ValidationStore) -> Optional[str]:
    if store.n == 0:
        print("[SAVE] nothing to save")
        return None

    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULT_DIR, f"validation_camera_only_{timestamp}.csv")
    stats = store.stats()

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "pixel_u",
                "pixel_v",
                "pred_x_mm",
                "pred_y_mm",
                "actual_x_mm",
                "actual_y_mm",
                "dx_mm",
                "dy_mm",
                "error_mm",
            ]
        )
        for i, r in enumerate(store.records, 1):
            writer.writerow(
                [
                    i,
                    r.pixel_u,
                    r.pixel_v,
                    r.pred_x,
                    r.pred_y,
                    r.actual_x,
                    r.actual_y,
                    r.dx,
                    r.dy,
                    r.error,
                ]
            )

        writer.writerow([])
        writer.writerow(["mean_error_mm", stats["mean"]])
        writer.writerow(["rmse_mm", stats["rmse"]])
        writer.writerow(["max_error_mm", stats["max"]])
        writer.writerow(["std_mm", stats["std"]])
        writer.writerow(["timestamp", datetime.now().isoformat()])

    print(f"[SAVE] {path}")
    return path


def draw(frame: np.ndarray, store: ValidationStore, status: str, last_click=None) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    for i, r in enumerate(store.records, 1):
        u, v = int(r.pixel_u), int(r.pixel_v)
        cv2.circle(img, (u, v), 7, (0, 180, 255), -1)
        cv2.circle(img, (u, v), 10, (255, 255, 255), 1)
        cv2.putText(
            img,
            f"V{i} {r.error:.1f}mm",
            (u + 12, v - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if last_click is not None:
        cv2.drawMarker(
            img,
            (int(last_click[0]), int(last_click[1])),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            24,
            2,
            cv2.LINE_AA,
        )

    canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :w] = img

    panel_x = w + 8
    cv2.rectangle(canvas, (w, 0), (w + PANEL_W, h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (w, 0), (w + PANEL_W, 36), (28, 28, 155), -1)
    cv2.putText(
        canvas,
        "VALIDATION",
        (panel_x, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Pts: {store.n}",
        (panel_x, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    stats = store.stats()
    if stats:
        cv2.putText(
            canvas,
            "Error stats",
            (panel_x, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 220, 100),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"mean : {stats['mean']:.1f} mm",
            (panel_x, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"RMSE : {stats['rmse']:.1f} mm",
            (panel_x, 133),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"max  : {stats['max']:.1f} mm",
            (panel_x, 151),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"std  : {stats['std']:.1f} mm",
            (panel_x, 169),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "No points yet",
            (panel_x, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (w, h - 80), (w + PANEL_W, h - 52), (38, 38, 38), -1)
    cv2.putText(
        canvas,
        "Status:",
        (panel_x, h - 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    for line_idx, chunk in enumerate([status[i:i + 28] for i in range(0, min(len(status), 56), 28)]):
        cv2.putText(
            canvas,
            chunk,
            (panel_x, h - 48 + line_idx * 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (0, 215, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (w, h - 50), (w + PANEL_W, h), (18, 18, 18), -1)
    keys = "Click=add  P=print  S=save  Z=undo  R=reset  Q=quit"
    cv2.putText(canvas, keys[:38], (panel_x, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
    cv2.putText(canvas, keys[38:], (panel_x, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

    return canvas


def main():
    print("=" * 70)
    print("Camera-only Calibration Validation")
    print("=" * 70)

    H = load_homography()
    cam = open_camera()
    store = ValidationStore()
    status = "Click a known point in the camera image"
    last_click = None

    win = "Calibration Validation"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    print("\n[READY]")
    print("  Click a point in the image, then enter the real robot XY in the terminal.")
    print("  Press P to print summary, S to save, Q/ESC to quit.\n")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                cv2.waitKey(30)
                continue
            global camera_width
            if camera_width == 0:
                camera_width = frame.shape[1]

            cv2.imshow(win, draw(frame, store, status, last_click))
            key = cv2.waitKey(1) & 0xFF

            if clicks:
                u, v = clicks.pop(0)
                last_click = (u, v)
                pred = pixel_to_robot(H, u, v)

                frozen = draw(frame, store, "Enter actual robot XY in terminal", last_click)
                cv2.imshow(win, frozen)
                cv2.waitKey(1)

                actual = ask_actual_xy((u, v), pred)
                if actual is None:
                    status = "Point skipped"
                    continue

                record = ValidationRecord(
                    pixel_u=float(u),
                    pixel_v=float(v),
                    pred_x=pred[0],
                    pred_y=pred[1],
                    actual_x=actual[0],
                    actual_y=actual[1],
                )
                store.add(record)
                print_record(record, store.n)
                status = f"V{store.n} added | error={record.error:.2f} mm"

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("p"), ord("P")):
                print_summary(store)

            elif key in (ord("s"), ord("S")):
                save_csv(store)
                status = "Saved CSV"

            elif key in (ord("z"), ord("Z")):
                store.undo()
                status = f"Undo | n={store.n}"
                last_click = None

            elif key in (ord("r"), ord("R")):
                ans = input("[RESET] Delete all validation points? Type 'yes': ").strip().lower()
                if ans == "yes":
                    store.reset()
                    last_click = None
                    status = "Reset complete"

    finally:
        if store.n > 0:
            print_summary(store)
            ans = input("Save CSV before exit? (y/n): ").strip().lower()
            if ans == "y":
                save_csv(store)

        cam.release()
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    main()
