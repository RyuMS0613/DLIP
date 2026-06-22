#!/usr/bin/env python3
"""
Camera-to-Robot Homography Calibration  (Steps 1–7)

Workflow:
  1. Camera fixed overhead (Spinnaker SDK or USB webcam)
  2. Workspace plane at fixed Z — table surface
  3. Place physical markerㅜ at known robot positions across workspace
  4. Click marker in image     → pixel (u, v) recorded
  5. Enter / auto-read robot XY → robot (X, Y) mm recorded
  6. Press H                   → homography computed (RANSAC, min 4 pts)
  7. Press T                   → test mode: click any point → predicted robot XY

Optional robot server (WSL):
  Run robot_calib_server.py in WSL to enable automatic robot movement.
  Without it, enter robot XY manually in the terminal.

Keys:
  Click       Add calibration point (calib mode) / predict XY (test mode)
  H           Compute homography
  T           Switch to TEST mode
  C           Switch to CALIBRATE mode
  G           Grid auto-calibration (requires robot server)
  S           Save calibration data + homography.npy
  Z           Undo last point
  P           Print verification table
  R           Reset all data
  Q / ESC     Quit
"""

import cv2
import numpy as np
import json
import os
import socket
import time
from datetime import datetime
from typing import Optional, List, Tuple

# ── PySpin (Spinnaker SDK) ────────────────────────────────────
try:
    import PySpin
    _PYSPIN = True
except ImportError:
    _PYSPIN = False

# ── File paths ────────────────────────────────────────────────
_DIR           = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON     = os.path.join(_DIR, "calib_data.json")
HOMOGRAPHY_NPY = os.path.join(_DIR, "homography.npy")

# ── Calibration settings ──────────────────────────────────────
MIN_PTS  = 4    # minimum points required for homography
REC_PTS  = 9    # recommended (3×3 grid)

# Fixed Z of calibration plane [mm] — height of workspace table surface
CALIB_Z_MM = 0.0   # ← adjust to your actual workspace height

# Predefined 3×3 grid for auto-calibration mode (robot XY in mm)
# Adjust these to cover your workspace evenly
GRID_POSITIONS: List[Tuple[float, float]] = [
    (200, 0), (200, 100), (200, 200),
    (300, 0), (300, 100), (300, 200),
    (400, 0), (400, 100), (400, 200),
]

# ── Robot TCP server ──────────────────────────────────────────
ROBOT_HOST = "127.0.0.1"   # WSL is reachable via localhost on Windows 11
ROBOT_PORT = 9999

# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

class _SpinnakerCam:
    def __init__(self):
        self._sys = PySpin.System.GetInstance()
        cam_list = self._sys.GetCameras()
        if cam_list.GetSize() == 0:
            self._sys.ReleaseInstance()
            raise RuntimeError("No Spinnaker camera found")
        self._cam = cam_list.GetByIndex(0)
        self._cam.Init()
        self._cam.BeginAcquisition()
        self._processor = PySpin.ImageProcessor()
        self._processor.SetColorProcessing(
            PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    def read(self) -> Optional[np.ndarray]:
        img = self._cam.GetNextImage(1000)
        if img.IsIncomplete():
            img.Release()
            return None
        bgr = self._processor.Convert(img, PySpin.PixelFormat_BGR8).GetNDArray().copy()
        img.Release()
        return bgr

    def release(self):
        self._cam.EndAcquisition()
        self._cam.DeInit()
        del self._cam
        self._sys.ReleaseInstance()


class _OpenCVCam:
    def __init__(self, idx: int = 0):
        self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {idx}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        self._cap.release()


def open_camera():
    if _PYSPIN:
        try:
            cam = _SpinnakerCam()
            print("[CAM] Spinnaker camera OK")
            return cam
        except Exception as e:
            print(f"[CAM] Spinnaker failed ({e}), falling back to OpenCV")
    cam = _OpenCVCam()
    print("[CAM] OpenCV VideoCapture OK")
    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Robot TCP communication  (optional)
# ─────────────────────────────────────────────────────────────────────────────

_rsock: Optional[socket.socket] = None


def robot_connect() -> bool:
    global _rsock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((ROBOT_HOST, ROBOT_PORT))
        _rsock = s
        print(f"[ROBOT] Connected to server {ROBOT_HOST}:{ROBOT_PORT}")
        return True
    except Exception as e:
        print(f"[ROBOT] Not connected ({e})  →  manual XY input mode")
        return False


def robot_cmd(cmd: str) -> Optional[str]:
    if _rsock is None:
        return None
    try:
        _rsock.sendall((cmd.strip() + "\n").encode())
        return _rsock.recv(512).decode().strip()
    except Exception as e:
        print(f"[ROBOT] comm error: {e}")
        return None


def robot_move(x: float, y: float, z: float = CALIB_Z_MM) -> bool:
    resp = robot_cmd(f"MOVE {x:.2f} {y:.2f} {z:.2f}")
    return resp is not None and resp.startswith("OK")


def robot_get_pos() -> Optional[Tuple[float, float]]:
    """Return current robot (X, Y) from server, or None."""
    resp = robot_cmd("GET_POS")
    if resp and resp.startswith("POS"):
        parts = resp.split()
        if len(parts) >= 3:
            return float(parts[1]), float(parts[2])
    return None


def robot_home() -> bool:
    resp = robot_cmd("HOME")
    return resp is not None and resp.startswith("OK")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data + homography
# ─────────────────────────────────────────────────────────────────────────────

class Calibration:
    def __init__(self):
        self.px: List[Tuple[float, float]] = []   # pixel (u, v)
        self.rb: List[Tuple[float, float]] = []   # robot (X, Y) mm
        self.H:  Optional[np.ndarray]      = None
        self.mode: str = "calib"                  # "calib" | "test"

    @property
    def n(self) -> int:
        return len(self.px)

    def add(self, px: Tuple[float, float], rb: Tuple[float, float]):
        self.px.append(tuple(px))
        self.rb.append(tuple(rb))

    def undo(self):
        if self.px:
            self.px.pop()
            self.rb.pop()
            if self.n < MIN_PTS:
                self.H = None

    # ── Step 6: Compute homography ────────────────────────────
    def compute_H(self) -> Tuple[bool, dict]:
        if self.n < MIN_PTS:
            return False, {"err": f"Need ≥{MIN_PTS} pts (have {self.n})"}
        src = np.float32(self.px)
        dst = np.float32(self.rb)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return False, {"err": "findHomography failed (collinear points?)"}
        self.H = H
        return True, self._stats()

    def predict(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        if self.H is None:
            return None
        pt  = cv2.perspectiveTransform(np.float32([[[u, v]]]), self.H)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def _errors(self) -> List[float]:
        if self.H is None:
            return []
        errs = []
        for (u, v), (rx, ry) in zip(self.px, self.rb):
            px2, py2 = self.predict(u, v)
            errs.append(float(np.hypot(px2 - rx, py2 - ry)))
        return errs

    def _stats(self) -> dict:
        errs = self._errors()
        if not errs:
            return {}
        a = np.array(errs)
        return {
            "mean": float(np.mean(a)),
            "rmse": float(np.sqrt(np.mean(a ** 2))),
            "max":  float(np.max(a)),
        }

    # ── Persist ───────────────────────────────────────────────
    def save(self):
        data = {
            "px_pts":     self.px,
            "rb_pts":     self.rb,
            "calib_z_mm": CALIB_Z_MM,
            "timestamp":  datetime.now().isoformat(),
        }
        with open(CALIB_JSON, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SAVE] {CALIB_JSON}")
        if self.H is not None:
            np.save(HOMOGRAPHY_NPY, self.H)
            print(f"[SAVE] {HOMOGRAPHY_NPY}")

    def load(self) -> bool:
        if not os.path.exists(CALIB_JSON):
            return False
        with open(CALIB_JSON) as f:
            d = json.load(f)
        self.px = [tuple(p) for p in d["px_pts"]]
        self.rb = [tuple(p) for p in d["rb_pts"]]
        print(f"[LOAD] {self.n} calibration points")
        if os.path.exists(HOMOGRAPHY_NPY):
            self.H = np.load(HOMOGRAPHY_NPY)
            print(f"[LOAD] homography.npy  {self._stats()}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# UI drawing
# ─────────────────────────────────────────────────────────────────────────────

PANEL_W = 280  # right-side info panel width

def draw(frame: np.ndarray, cal: Calibration,
         status: str, crosshair: Optional[Tuple] = None) -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]
    errs = cal._errors()

    # Calibration points
    for i, ((u, v), (rx, ry)) in enumerate(zip(cal.px, cal.rb)):
        u, v = int(u), int(v)
        clr = (0, 210, 70) if cal.H is not None else (0, 160, 255)
        cv2.circle(img, (u, v), 7, clr, -1)
        cv2.circle(img, (u, v), 9, (255, 255, 255), 1)
        lbl = f"P{i+1}"
        if cal.H is not None and i < len(errs):
            lbl += f" {errs[i]:.1f}mm"
        cv2.putText(img, lbl, (u + 11, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # Crosshair at last click
    if crosshair:
        cx, cy = int(crosshair[0]), int(crosshair[1])
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)

    # Right-side panel
    canvas = np.zeros((H, W + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :W] = img

    px = W + 8  # text x inside panel
    hdr_clr = (28, 28, 155) if cal.mode == "calib" else (100, 18, 110)
    cv2.rectangle(canvas, (W, 0), (W + PANEL_W, H), (30, 30, 30), -1)

    # Mode badge
    cv2.rectangle(canvas, (W, 0), (W + PANEL_W, 36), hdr_clr, -1)
    mode_txt = "CALIBRATE" if cal.mode == "calib" else "TEST"
    cv2.putText(canvas, mode_txt, (px, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Points count
    cv2.putText(canvas, f"Pts: {cal.n} / {REC_PTS}", (px, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Homography stats
    s = cal._stats()
    if s:
        cv2.putText(canvas, "Homography", (px, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)
        cv2.putText(canvas, f"mean : {s['mean']:.1f} mm", (px, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)
        cv2.putText(canvas, f"RMSE : {s['rmse']:.1f} mm", (px, 133),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)
        cv2.putText(canvas, f"max  : {s['max']:.1f} mm", (px, 151),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)
    else:
        cv2.putText(canvas, "H: not computed", (px, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 100, 100), 1)

    # Status
    cv2.rectangle(canvas, (W, H - 80), (W + PANEL_W, H - 52), (38, 38, 38), -1)
    cv2.putText(canvas, "Status:", (px, H - 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
    # wrap status text across two lines if needed
    for li, chunk in enumerate([status[i:i+28] for i in range(0, min(len(status), 56), 28)]):
        cv2.putText(canvas, chunk, (px, H - 64 + 16 + li * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 215, 0), 1)

    # Key guide (vertical list)
    cv2.rectangle(canvas, (W, H - 50), (W + PANEL_W, H), (18, 18, 18), -1)
    if cal.mode == "calib":
        keys = ["Click=add", "H=homography", "T=test", "S=save", "Z=undo", "R=reset", "Q=quit"]
    else:
        keys = ["Click=predict", "C=calib", "S=save", "Q=quit"]
    key_txt = "  ".join(keys)
    cv2.putText(canvas, key_txt[:38], (px, H - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
    cv2.putText(canvas, key_txt[38:], (px, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

    return canvas


# ── Step 7: Verification table ────────────────────────────────
def print_table(cal: Calibration):
    errs = cal._errors()
    print("\n" + "=" * 76)
    print(f"  {'#':<3} {'Pixel (u,v)':<18} {'Robot actual (mm)':<22}"
          f" {'Predicted (mm)':<22} {'Err(mm)'}")
    print("-" * 76)
    for i, ((u, v), (rx, ry)) in enumerate(zip(cal.px, cal.rb)):
        if cal.H is not None and i < len(errs):
            px2, py2 = cal.predict(u, v)
            print(f"  P{i+1:<2} ({u:>6.0f},{v:>5.0f})  "
                  f" ({rx:>7.1f},{ry:>7.1f})    "
                  f" ({px2:>7.1f},{py2:>7.1f})    {errs[i]:>6.2f}")
        else:
            print(f"  P{i+1:<2} ({u:>6.0f},{v:>5.0f})  "
                  f" ({rx:>7.1f},{ry:>7.1f})    {'--':>20}  --")
    if errs:
        s = cal._stats()
        print("-" * 76)
        print(f"  {'Mean error':>52}: {s['mean']:.2f} mm")
        print(f"  {'RMSE':>52}: {s['rmse']:.2f} mm")
        print(f"  {'Max error':>52}: {s['max']:.2f} mm")
    print("=" * 76 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mouse callback
# ─────────────────────────────────────────────────────────────────────────────

_clicks: List[Tuple[int, int]] = []
_cam_width: int = 0  # set after first frame; clicks in panel area are ignored

def _on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if _cam_width == 0 or x < _cam_width:
            _clicks.append((x, y))


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ask_robot_xy(px_xy: Tuple) -> Optional[Tuple[float, float]]:
    print(f"\n  Pixel clicked: ({px_xy[0]:.0f}, {px_xy[1]:.0f})")
    raw = input("  Robot XY [mm]  e.g. '150 200'  (Enter = skip): ").strip()
    if not raw:
        return None
    try:
        parts = raw.replace(",", " ").split()
        return float(parts[0]), float(parts[1])
    except Exception:
        print("  Invalid input. Point skipped.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Grid auto-calibration mode
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_calibration(cal: Calibration, cam, WIN: str) -> int:
    """
    Moves robot to each predefined grid position, waits for user to click
    on the marker in the image, then records the pixel↔robot pair.
    Returns the number of new points added.
    """
    if _rsock is None:
        print("[GRID] Robot server not connected. Cannot run auto grid.")
        return 0

    added = 0
    print(f"\n[GRID] Starting auto-calibration with {len(GRID_POSITIONS)} grid positions")
    print(f"       Place a visible marker at each robot position and click it in the image.")
    print(f"       Press 'n' in terminal to skip a position.\n")

    for idx, (rx, ry) in enumerate(GRID_POSITIONS):
        print(f"\n[GRID] Moving to position {idx+1}/{len(GRID_POSITIONS)}: "
              f"X={rx}mm Y={ry}mm Z={CALIB_Z_MM}mm")
        ok = robot_move(rx, ry, CALIB_Z_MM)
        if not ok:
            print(f"  Robot move failed — skipping")
            continue

        print(f"  Robot arrived. Click the marker in the image (or 's' to skip).")
        skip = input("  Press Enter when ready, or type 's' to skip: ").strip().lower()
        if skip == "s":
            continue

        # Wait for user to click in image
        _clicks.clear()
        frame = None
        deadline = time.time() + 15.0   # 15 s timeout per position

        while time.time() < deadline:
            f = cam.read()
            if f is not None:
                frame = f
            if frame is None:
                cv2.waitKey(30)
                continue
            overlay = draw(frame, cal,
                           f"GRID {idx+1}/{len(GRID_POSITIONS)}: "
                           f"click on marker at ({rx},{ry})mm", None)
            cv2.rectangle(overlay, (0, 36), (frame.shape[1], 56), (0, 80, 0), -1)
            cv2.putText(overlay,
                        f"  Click the robot marker  |  robot pos = ({rx},{ry})mm",
                        (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.imshow(WIN, overlay)
            cv2.waitKey(30)

            if _clicks:
                cx, cy = _clicks.pop(0)
                cal.add((cx, cy), (rx, ry))
                added += 1
                print(f"  → Recorded P{cal.n}: pixel({cx},{cy}) ↔ robot({rx},{ry})mm")
                break
        else:
            print(f"  Timeout — skipping position {idx+1}")

    print(f"\n[GRID] Done. Added {added} new points. Total: {cal.n}")
    robot_home()
    return added


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Camera → Robot  Homography Calibration  [Steps 1–7]")
    print("=" * 60)

    # Step 1 – fixed camera
    cam = open_camera()

    cal = Calibration()

    if os.path.exists(CALIB_JSON):
        if input("\nPrevious calibration found. Load? (y/n): ").strip().lower() == "y":
            cal.load()

    # Optional robot server
    robot_ok = robot_connect()

    # OpenCV window
    WIN = "Homography Calibration"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, _on_mouse)

    print(f"\n[READY]")
    print(f"  Robot server : {'connected' if robot_ok else 'not connected (manual input)'}")
    print(f"  Loaded points: {cal.n}")
    print(f"  Calib Z (mm) : {CALIB_Z_MM}")
    print(f"  Min pts for H: {MIN_PTS}  |  Recommended: {REC_PTS}")
    print()

    status    = "Click on calibration marker to add point"
    crosshair: Optional[Tuple] = None
    frame:     Optional[np.ndarray] = None

    while True:
        f = cam.read()
        if f is not None:
            frame = f
            global _cam_width
            if _cam_width == 0:
                _cam_width = f.shape[1]

        if frame is None:
            cv2.waitKey(30)
            continue

        cv2.imshow(WIN, draw(frame, cal, status, crosshair))
        key = cv2.waitKey(1) & 0xFF

        # ── Process mouse click ───────────────────────────────
        if _clicks:
            cx, cy = _clicks.pop(0)
            crosshair = (cx, cy)

            if cal.mode == "calib":
                # Freeze window while terminal input
                frozen = draw(frame, cal,
                              f"Clicked ({cx},{cy}) — enter robot XY in terminal ↓",
                              crosshair)
                cv2.imshow(WIN, frozen)
                cv2.waitKey(1)

                rb_xy: Optional[Tuple[float, float]] = None

                if robot_ok:
                    rb_xy = robot_get_pos()
                    if rb_xy:
                        print(f"\n  [AUTO] Robot pos = ({rb_xy[0]:.1f}, {rb_xy[1]:.1f}) mm")

                if rb_xy is None:
                    rb_xy = _ask_robot_xy((cx, cy))

                if rb_xy:
                    cal.add((cx, cy), rb_xy)
                    status = (f"P{cal.n} added  "
                              f"px({cx},{cy}) → rb({rb_xy[0]:.0f},{rb_xy[1]:.0f})mm")
                    print(f"  → P{cal.n} saved.  Total: {cal.n} pts")
                else:
                    crosshair = None

            elif cal.mode == "test":
                if cal.H is None:
                    status = "No homography — press H first"
                else:
                    pred = cal.predict(cx, cy)
                    print(f"\n[TEST]  Pixel ({cx},{cy})  →  "
                          f"Robot ({pred[0]:.2f}, {pred[1]:.2f}) mm")
                    status = f"TEST ({cx},{cy}) → ({pred[0]:.1f},{pred[1]:.1f}) mm"

                    # Optionally move robot to verify
                    if robot_ok:
                        ans = input(
                            f"  Move robot to predicted ({pred[0]:.1f},{pred[1]:.1f})mm? (y/n): "
                        ).strip().lower()
                        if ans == "y":
                            ok = robot_move(pred[0], pred[1])
                            print(f"  Robot move: {'OK' if ok else 'FAILED'}")

                    # Measure actual error
                    raw = input(
                        "  Actual robot XY [mm] for error check (Enter = skip): "
                    ).strip()
                    if raw:
                        try:
                            parts = raw.replace(",", " ").split()
                            ax, ay = float(parts[0]), float(parts[1])
                            err = float(np.hypot(pred[0] - ax, pred[1] - ay))
                            status = (f"TEST err={err:.2f}mm  "
                                      f"pred({pred[0]:.1f},{pred[1]:.1f}) "
                                      f"actual({ax:.1f},{ay:.1f})")
                            print(f"  → Error: {err:.2f} mm")
                        except Exception:
                            pass

        # ── Key handling ──────────────────────────────────────
        if key in (ord("q"), 27):                     # Q / ESC — quit
            break

        elif key in (ord("h"), ord("H")):             # H — compute homography
            ok2, info = cal.compute_H()
            if ok2:
                s = info
                status = (f"H computed  "
                          f"mean={s['mean']:.1f} RMSE={s['rmse']:.1f} max={s['max']:.1f} mm")
                print(f"\n[HOMOGRAPHY]  mean={s['mean']:.2f}  "
                      f"RMSE={s['rmse']:.2f}  max={s['max']:.2f} mm")
                print_table(cal)
            else:
                status = f"H failed: {info.get('err')}"
                print(f"[ERROR] {info.get('err')}")

        elif key in (ord("t"), ord("T")):             # T — test mode
            if cal.H is None:
                status = "Compute H first (press H)"
            else:
                cal.mode = "test"
                status = "TEST mode — click to predict robot XY"
                print("[TEST MODE] Click any point in the image.")

        elif key in (ord("c"), ord("C")):             # C — calibrate mode
            cal.mode = "calib"
            status = "CALIBRATE mode"

        elif key in (ord("g"), ord("G")):             # G — grid auto-calibration
            if not robot_ok:
                print("[GRID] Robot server not connected.")
                status = "Grid mode requires robot server connection"
            else:
                n_added = run_grid_calibration(cal, cam, WIN)
                status = f"Grid done — {n_added} pts added, total: {cal.n}"

        elif key in (ord("s"), ord("S")):             # S — save
            if cal.H is None and cal.n >= MIN_PTS:
                cal.compute_H()
            cal.save()
            status = "Saved!"

        elif key in (ord("z"), ord("Z")):             # Z — undo
            cal.undo()
            status = f"Undo — pts remaining: {cal.n}"
            crosshair = None

        elif key in (ord("p"), ord("P")):             # P — print table
            print_table(cal)

        elif key in (ord("r"), ord("R")):             # R — reset
            ans = input("[RESET] Delete all calibration data? (yes): ").strip()
            if ans == "yes":
                cal = Calibration()
                crosshair = None
                status = "Reset complete"

    # ── Exit ─────────────────────────────────────────────────
    if cal.n > 0:
        if input("\nSave before exit? (y/n): ").strip().lower() == "y":
            if cal.H is None and cal.n >= MIN_PTS:
                cal.compute_H()
            cal.save()

    if robot_ok and _rsock:
        try:
            robot_cmd("QUIT")
            _rsock.close()
        except Exception:
            pass

    cam.release()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()
