#!/usr/bin/env python3
"""
Final_Camera.py  (Windows 실행)

YOLO OBB 검출 → 숫자 선택 → 좌표 계산 → WSL Final_Robot.py로 1회 전송
→ 로봇 자율 실행 → done/failed 수신 → 재검출 반복

통신: Camera→Robot (execute 1회) / Robot→Camera (done or failed 1회)

키 (카메라 창):
  Space  : 검출 스냅샷
  1~9    : 물체 선택
  H      : 홈 복귀
  Q/ESC  : 종료
"""

import cv2
import json
import math
import msvcrt
import numpy as np
import os
import socket
import sys
import threading
import time
from typing import List, Optional, Tuple

from ultralytics import YOLO

try:
    import PySpin
    _PYSPIN = True
except ImportError:
    _PYSPIN = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ★ 클래스별 작업 파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Z_SCALPEL = 135
Z_BOTTLE  = 100

BOTTLE_X_OFFSET          = -120.0
APPROACH_DZ_SCALPEL      = 60.0
APPROACH_X_OFFSET_BOTTLE = -50.0

CLASS_CONFIG: dict = {
    "Blade":     dict(z_mm=Z_SCALPEL, roll=0.0, pitch=180.0,
                      use_yaw=True,  fixed_yaw=0.0,
                      x_offset=0.0, y_offset=0.0, yaw_offset=0.0,
                      approach_dz=APPROACH_DZ_SCALPEL),
    "scalpel10": dict(z_mm=Z_SCALPEL, roll=0.0, pitch=180.0,
                      use_yaw=True,  fixed_yaw=0.0,
                      x_offset=0.0, y_offset=0.0, yaw_offset=0.0,
                      approach_dz=APPROACH_DZ_SCALPEL),
    "scalpel11": dict(z_mm=Z_SCALPEL, roll=0.0, pitch=180.0,
                      use_yaw=True,  fixed_yaw=0.0,
                      x_offset=0.0, y_offset=0.0, yaw_offset=0.0,
                      approach_dz=APPROACH_DZ_SCALPEL),
    "scalpel15": dict(z_mm=Z_SCALPEL, roll=0.0, pitch=180.0,
                      use_yaw=True,  fixed_yaw=0.0,
                      x_offset=0.0, y_offset=0.0, yaw_offset=0.0,
                      approach_dz=APPROACH_DZ_SCALPEL),
    "bottle":    dict(z_mm=Z_BOTTLE,  roll=0.0, pitch=120.0,
                      use_yaw=False, fixed_yaw=0.0,
                      x_offset=BOTTLE_X_OFFSET, y_offset=0.0, yaw_offset=0.0,
                      approach_dz=0.0,
                      approach_x_offset=APPROACH_X_OFFSET_BOTTLE,
                      radial_offset=True),
}

DEFAULT_CONFIG = dict(z_mm=140.0, roll=0.0, pitch=180.0,
                      use_yaw=True, fixed_yaw=0.0,
                      x_offset=0.0, y_offset=0.0, yaw_offset=0.0,
                      approach_dz=60.0)

YAW_OFFSET_GLOBAL = -3.0
CONF_THRES        = 0.75
MAX_REACH_MM      = 627.0
MIN_Z_MM          = 40

PLACE_JOINTS_BASE = [
    math.radians(  0.0),
    math.radians( 95.9),
    math.radians(-32.8),
    math.radians(  0.1),
    math.radians(-58.2),
    math.radians(  0.0),
]
PLACE_J6_NORMAL  = math.radians( 90.0)
PLACE_J6_FLIPPED = math.radians(-90.0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ★ Blade 방향 보정 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCALPEL_CLASSES              = {"scalpel10", "scalpel11", "scalpel15"}
BLADE_TARGET_PIXEL_ANGLE_DEG = 0.0
BLADE_IN_OBB_MARGIN          = 0.55

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MAIN_DIR      = os.path.dirname(os.path.abspath(__file__))
_FINAL_DIR     = os.path.dirname(_MAIN_DIR)
MODEL_PATH     = os.path.join(_FINAL_DIR, "Model",       "weights", "best.pt")
HOMOGRAPHY_NPY = os.path.join(_FINAL_DIR, "Calibration", "homography.npy")

WSL_HOST = "localhost"
WSL_PORT = 7777
WIN      = "Final_Camera"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State 정의 — 인덱스 상수 + 이름/색상 배열
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDLE, INPUT, PLAN, WAITING = range(4)

STATE_NAMES = [
    "IDLE",     # 0  실시간 YOLO 검출 표시 + Space로 선택 진입
    "INPUT",    # 1  숫자 키로 물체 선택
    "PLAN",     # 2  좌표 계산 + 로봇에 전송
    "WAITING",  # 3  로봇 자율 실행 중 대기
]

STATE_COLORS = [
    (0,   180,   0),   # IDLE
    (0,   220, 220),   # INPUT
    (0,   200, 255),   # PLAN
    (200, 160,   0),   # WAITING
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  앱 컨텍스트 — 모든 상태 핸들러가 공유하는 데이터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AppCtx:
    def __init__(self, cam, model, H_mat, rsock, wsl_queue, wsl_lock):
        # 공유 리소스
        self.cam       = cam
        self.model     = model
        self.H_mat     = H_mat
        self.rsock     = rsock
        self.wsl_queue = wsl_queue
        self.wsl_lock  = wsl_lock
        # 가변 상태
        self.state             = IDLE
        self.frame             = None          # 최신 카메라 프레임
        self.detected_snapshot = []            # DETECT 시점 스냅샷
        self.current_det       = None          # 선택된 검출 결과
        self.status_msg        = ""            # 화면 하단 메시지
        self.should_quit       = False         # True → 메인 루프 종료

    def grab_frame(self) -> np.ndarray:
        """카메라에서 프레임 읽기. 실패 시 이전 프레임 또는 검정 화면 반환."""
        f = self.cam.read()
        if f is not None:
            self.frame = f
        return self.frame if self.frame is not None else np.zeros((480, 640, 3), np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  카메라
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _SpinnakerCam:
    def __init__(self):
        self._sys = PySpin.System.GetInstance()
        cam_list  = self._sys.GetCameras()
        if cam_list.GetSize() == 0:
            self._sys.ReleaseInstance()
            raise RuntimeError("Spinnaker 카메라 없음")
        self._cam  = cam_list.GetByIndex(0)
        self._cam.Init()
        self._cam.BeginAcquisition()
        self._proc = PySpin.ImageProcessor()
        self._proc.SetColorProcessing(
            PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    def read(self) -> Optional[np.ndarray]:
        img = self._cam.GetNextImage(1000)
        if img.IsIncomplete():
            img.Release()
            return None
        bgr = self._proc.Convert(img, PySpin.PixelFormat_BGR8).GetNDArray().copy()
        img.Release()
        return bgr

    def release(self):
        self._cam.EndAcquisition()
        self._cam.DeInit()
        del self._cam
        self._sys.ReleaseInstance()


class _OpenCVCam:
    def __init__(self, idx: int = 0):
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"카메라 {idx} 열기 실패")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        self._cap.release()


def open_camera():
    if _PYSPIN:
        try:
            cam = _SpinnakerCam()
            print("[CAM] Spinnaker OK")
            return cam
        except Exception as e:
            print(f"[CAM] Spinnaker 실패 ({e}), OpenCV로 전환")
    cam = _OpenCVCam()
    print("[CAM] OpenCV OK")
    return cam


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  YOLO / Homography / 좌표 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def load_homography() -> np.ndarray:
    if not os.path.exists(HOMOGRAPHY_NPY):
        raise FileNotFoundError(f"homography.npy 없음: {HOMOGRAPHY_NPY}")
    return np.load(HOMOGRAPHY_NPY)


def pixel_to_robot(H: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    pt  = np.float32([[[px, py]]])
    res = cv2.perspectiveTransform(pt, H)
    return float(res[0, 0, 0]), float(res[0, 0, 1])


def obb_angle_to_yaw(angle_rad: float, yaw_offset: float = 0.0) -> float:
    yaw = -(math.degrees(angle_rad) + YAW_OFFSET_GLOBAL + yaw_offset)
    yaw = (yaw + 180.0) % 360.0 - 180.0
    if yaw > 90.0:   yaw -= 180.0
    elif yaw < -90.0: yaw += 180.0
    return yaw


def point_in_obb(px, py, cx, cy, w, h, angle_rad) -> bool:
    dx, dy  = px - cx, py - cy
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    local_x =  dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    return (abs(local_x) <= w * BLADE_IN_OBB_MARGIN and
            abs(local_y) <= h * BLADE_IN_OBB_MARGIN)


def find_blade_for_scalpel(scalpel: dict, all_dets: List[dict]) -> Optional[dict]:
    best, best_conf = None, -1.0
    for d in all_dets:
        if d["cls_name"] != "Blade":
            continue
        if point_in_obb(d["cx"], d["cy"],
                        scalpel["cx"], scalpel["cy"],
                        scalpel["w"],  scalpel["h"],
                        scalpel["angle_rad"]):
            if d["conf"] > best_conf:
                best, best_conf = d, d["conf"]
    return best


def obb_angle_to_yaw_with_blade(scalpel: dict, blade: dict,
                                 yaw_offset: float = 0.0) -> Tuple[float, bool]:
    angle_rad = scalpel["angle_rad"]
    dvx = blade["cx"] - scalpel["cx"]
    dvy = blade["cy"] - scalpel["cy"]
    ax, ay   = math.cos(angle_rad), math.sin(angle_rad)
    dot      = dvx * ax + dvy * ay
    blade_ax = ax if dot >= 0 else -ax
    blade_ay = ay if dot >= 0 else -ay
    target_rad = math.radians(BLADE_TARGET_PIXEL_ANGLE_DEG)
    alignment  = blade_ax * math.cos(target_rad) + blade_ay * math.sin(target_rad)
    return obb_angle_to_yaw(angle_rad, yaw_offset), (alignment < 0)


def run_yolo(model: YOLO, frame: np.ndarray) -> List[dict]:
    results = model(frame, conf=CONF_THRES, verbose=False)
    dets, r = [], results[0]
    if r.obb is None or len(r.obb) == 0:
        return dets
    xywhr   = r.obb.xywhr.cpu().numpy()
    confs   = r.obb.conf.cpu().numpy()
    clss    = r.obb.cls.cpu().numpy()
    corners = r.obb.xyxyxyxy.cpu().numpy().astype(np.int32)
    for i in range(len(xywhr)):
        cx, cy, w, h, angle_rad = xywhr[i]
        dets.append(dict(cx=float(cx), cy=float(cy), w=float(w), h=float(h),
                         angle_rad=float(angle_rad), conf=float(confs[i]),
                         cls_name=r.names.get(int(clss[i]), str(int(clss[i]))),
                         corners=corners[i]))
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


def is_reachable(x_mm, y_mm, z_mm) -> Tuple[bool, str]:
    dist = math.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
    if dist > MAX_REACH_MM:
        return False, f"반경 {dist:.1f}mm > 최대 {MAX_REACH_MM}mm"
    if z_mm < MIN_Z_MM:
        return False, f"Z={z_mm:.1f}mm < 하한 {MIN_Z_MM}mm"
    return True, ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Action Plan (좌표 계산)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def action_plan(det: dict, H_mat: np.ndarray,
                all_dets: List[dict]) -> Optional[dict]:
    cfg           = CLASS_CONFIG.get(det["cls_name"], DEFAULT_CONFIG)
    rx, ry        = pixel_to_robot(H_mat, det["cx"], det["cy"])
    blade_flipped = False

    if cfg.get("radial_offset", False):
        theta = math.atan2(ry, rx)
        x_off = cfg.get("x_offset", 0.0)
        rx   += x_off * math.cos(theta)
        ry   += x_off * math.sin(theta)
        yaw   = math.degrees(theta)
    else:
        rx += cfg["x_offset"]
        ry += cfg["y_offset"]
        if cfg["use_yaw"]:
            if det["cls_name"] in SCALPEL_CLASSES:
                blade = find_blade_for_scalpel(det, all_dets)
                if blade:
                    yaw, blade_flipped = obb_angle_to_yaw_with_blade(det, blade, cfg["yaw_offset"])
                    print(f"  [PLAN] Blade 검출  Yaw={yaw:.1f}°  blade_flipped={blade_flipped}")
                else:
                    yaw = obb_angle_to_yaw(det["angle_rad"], cfg["yaw_offset"])
                    print(f"  [PLAN] Blade 미검출 — 기본 Yaw={yaw:.1f}°")
            else:
                yaw = obb_angle_to_yaw(det["angle_rad"], cfg["yaw_offset"])
        else:
            yaw = cfg["fixed_yaw"]

    ok, reason = is_reachable(rx, ry, cfg["z_mm"])
    if not ok:
        print(f"  [PLAN] 목표 거부: {reason}")
        return None

    print(f"  [PLAN] {det['cls_name']}  "
          f"X={rx:.1f} Y={ry:.1f} Z={cfg['z_mm']:.1f}mm  "
          f"P={cfg['pitch']:.0f} Yaw={yaw:.1f}°  blade_flipped={blade_flipped}")
    return dict(x_mm=rx, y_mm=ry, z_mm=cfg["z_mm"],
                roll=cfg["roll"], pitch=cfg["pitch"], yaw=yaw,
                cls_name=det["cls_name"], blade_flipped=blade_flipped)


def action_plan_approach(target: dict) -> dict:
    cfg          = CLASS_CONFIG.get(target["cls_name"], DEFAULT_CONFIG)
    approach_off = cfg.get("approach_x_offset", 0.0)
    if cfg.get("radial_offset", False):
        theta = math.atan2(target["y_mm"], target["x_mm"])
        ax    = target["x_mm"] + approach_off * math.cos(theta)
        ay    = target["y_mm"] + approach_off * math.sin(theta)
    else:
        ax = target["x_mm"] + approach_off
        ay = target["y_mm"]
    return dict(x_mm=ax, y_mm=ay,
                z_mm=target["z_mm"] + cfg["approach_dz"],
                roll=target["roll"], pitch=target["pitch"],
                yaw=target["yaw"], cls_name=target["cls_name"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TCP 소켓
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RobotSocket:
    def __init__(self):
        self._sock = None
        self._file = None

    def connect(self, host: str, port: int,
                retries: int = 10, delay: float = 2.0) -> bool:
        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self._sock = s
                self._file = s.makefile("r")
                print(f"[SOCKET] WSL 연결 완료 ({host}:{port})")
                return True
            except ConnectionRefusedError:
                print(f"[SOCKET] 연결 시도 {i+1}/{retries} ... ({delay}s 후 재시도)")
                time.sleep(delay)
        return False

    def send(self, msg: dict):
        if self._sock:
            try:
                self._sock.sendall((json.dumps(msg) + "\n").encode())
            except Exception as e:
                print(f"[SOCKET] 전송 실패: {e}")

    def recv(self) -> Optional[dict]:
        if self._file:
            try:
                line = self._file.readline()
                if line:
                    return json.loads(line.strip())
            except Exception as e:
                print(f"[SOCKET] 수신 오류: {e}")
        return None

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._file = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_blade_arrow(img: np.ndarray, scalpel_det: dict, blade_det: dict) -> None:
    scx, scy = int(scalpel_det["cx"]), int(scalpel_det["cy"])
    bcx, bcy = int(blade_det["cx"]),   int(blade_det["cy"])
    cv2.arrowedLine(img, (scx, scy), (bcx, bcy), (0, 140, 255), 2, tipLength=0.3)
    cv2.circle(img, (bcx, bcy), 6, (0, 80, 255), -1)
    cv2.putText(img, "Blade", (bcx + 6, bcy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)
    target_rad = math.radians(BLADE_TARGET_PIXEL_ANGLE_DEG)
    ex = int(scx + 40 * math.cos(target_rad))
    ey = int(scy + 40 * math.sin(target_rad))
    cv2.arrowedLine(img, (scx, scy), (ex, ey), (0, 255, 80), 2, tipLength=0.4)


def draw_frame(frame: np.ndarray, dets: List[dict], H_mat: np.ndarray,
               state: int, status_msg: str = "",
               selected_det: Optional[dict] = None) -> np.ndarray:
    img      = frame.copy()
    fh, fw   = img.shape[:2]
    is_input = (state == INPUT)
    non_blade = [d for d in dets if d["cls_name"] != "Blade"]

    # Blade-Scalpel 매핑 (시각화용)
    blade_map: dict = {}
    for d in dets:
        if d["cls_name"] in SCALPEL_CLASSES:
            b = find_blade_for_scalpel(d, dets)
            if b is not None:
                blade_map[id(d)] = b

    for i, d in enumerate(dets):
        if d["cls_name"] == "Blade":
            cv2.polylines(img, [d["corners"]], True, (60, 80, 200), 1)
            continue

        selected = (selected_det is d)
        clr = ((0, 0, 255) if selected
               else ((0, 220, 220) if is_input
                     else ((0, 220, 80) if i == 0 else (200, 130, 0))))
        cv2.polylines(img, [d["corners"]], True, clr, 2)
        cx, cy = int(d["cx"]), int(d["cy"])
        cv2.circle(img, (cx, cy), 5, clr, -1)
        rx, ry = pixel_to_robot(H_mat, d["cx"], d["cy"])

        blade = blade_map.get(id(d))
        if d["cls_name"] in SCALPEL_CLASSES and blade is not None:
            yaw, blade_flipped = obb_angle_to_yaw_with_blade(d, blade)
            yaw_label = f"yaw={yaw:.0f}° {'↺' if blade_flipped else '✓'}"
            _draw_blade_arrow(img, d, blade)
        elif d["cls_name"] == "bottle":
            yaw_label = "(radial)"
        else:
            yaw = obb_angle_to_yaw(d["angle_rad"])
            yaw_label = f"yaw={yaw:.0f}°"

        nb_idx = non_blade.index(d) if d in non_blade else -1
        if is_input:
            name_tag = f"[{nb_idx+1}] {d['cls_name']}" if nb_idx >= 0 else d["cls_name"]
            cv2.putText(img, f"{name_tag} {d['conf']:.2f}",
                        (cx+8, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)
            cv2.putText(img, f"({rx:.0f},{ry:.0f})mm  {yaw_label}",
                        (cx+8, cy+8),  cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f"{d['cls_name']} {d['conf']:.2f}",
                        (cx+8, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        ((0, 0, 255) if selected else (255, 255, 255)), 1, cv2.LINE_AA)
            cv2.putText(img, f"({rx:.0f},{ry:.0f})mm  {yaw_label}",
                        (cx+8, cy+4),  cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 255, 180), 1, cv2.LINE_AA)

    # 상태 바
    s_clr = STATE_COLORS[state]
    cv2.rectangle(img, (0, fh-26), (fw, fh), (20, 20, 20), -1)

    if state == IDLE:
        bar = f"STATE: {'IDLE':<12}  Space=실행  H=홈  Q=종료"
    elif state == INPUT:
        hint = "  ".join(f"[{i+1}]{d['cls_name']}" for i, d in enumerate(non_blade))
        bar  = f"SELECT: {hint}   ESC=취소"
    elif state == WAITING:
        bar = f"STATE: {'WAITING':<12}  {status_msg}  (WSL 터미널에서 s=비상정지)"
    else:
        bar = f"STATE: {STATE_NAMES[state]:<12}  {status_msg}"

    cv2.putText(img, bar, (8, fh-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, s_clr, 1, cv2.LINE_AA)
    return img


def read_key(wait_ms: int = 30) -> int:
    key = cv2.waitKey(wait_ms) & 0xFF
    if key != 0xFF:
        return key
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch == b' ':            return 32
        if ch in (b'q', b'Q'):   return ord('q')
        if ch == b'\x1b':        return 27
        if ch in (b'h', b'H'):   return ord('h')
        if b'1' <= ch <= b'9':   return ch[0]
    return 0xFF


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WSL 수신 처리 (매 루프 최초 실행)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def recv_wsl(ctx: AppCtx) -> None:
    with ctx.wsl_lock:
        pending = list(ctx.wsl_queue)
        ctx.wsl_queue.clear()

    for m in pending:
        ws = m.get("status", "")
        if ws == "ready":
            print("[WSL] 준비 완료")
        elif ws == "done":
            print("[WSL] 작업 완료 → 검출 재시작\n")
            ctx.state      = IDLE
            ctx.status_msg = ""
            ctx.detected_snapshot = []
            ctx.current_det = None
        elif ws == "failed":
            print(f"[WSL] 실패 ({m.get('msg', '')}) → 검출 재시작\n")
            ctx.state      = IDLE
            ctx.status_msg = m.get("msg", "failed")
            ctx.detected_snapshot = []
            ctx.current_det = None
        elif ws == "estop":
            print("[WSL] 비상 정지 → 검출 재시작\n")
            ctx.state      = IDLE
            ctx.status_msg = "estop"
            ctx.detected_snapshot = []
            ctx.current_det = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  상태 핸들러
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def state_idle(ctx: AppCtx) -> None:
    """실시간 YOLO 검출 표시. Space → 현재 검출 스냅샷 저장 후 선택 / H → 홈 / Q → 종료"""
    frame = ctx.grab_frame()
    dets  = run_yolo(ctx.model, frame)
    cv2.imshow(WIN, draw_frame(frame, dets, ctx.H_mat, ctx.state, ctx.status_msg))
    key = read_key(30)

    if key == 32:
        non_blade = [d for d in dets if d["cls_name"] != "Blade"]
        blade_det = [d for d in dets if d["cls_name"] == "Blade"]
        print(f"\n[DETECT] {len(non_blade)}개 검출:")
        for i, d in enumerate(non_blade):
            rx, ry = pixel_to_robot(ctx.H_mat, d["cx"], d["cy"])
            print(f"  [{i+1}] {d['cls_name']:<12} conf={d['conf']:.2f}  "
                  f"px=({d['cx']:.0f},{d['cy']:.0f})  robot=({rx:.1f},{ry:.1f})mm")
        for d in blade_det:
            rx, ry = pixel_to_robot(ctx.H_mat, d["cx"], d["cy"])
            print(f"  [*] Blade(보조)     conf={d['conf']:.2f}  "
                  f"px=({d['cx']:.0f},{d['cy']:.0f})  robot=({rx:.1f},{ry:.1f})mm")
        if not non_blade:
            print("  → 선택 가능한 물체 없음\n")
        else:
            print("  → 숫자 키로 선택  ESC=취소\n")
            ctx.detected_snapshot = dets
            ctx.state = INPUT
    elif key in (ord('h'), ord('H')):
        ctx.rsock.send({"cmd": "home"})
        print("[→WSL] 홈 복귀 명령")
        ctx.state      = WAITING
        ctx.status_msg = "홈 복귀 중..."
    elif key in (ord('q'), 27):
        ctx.should_quit = True


def state_input(ctx: AppCtx) -> None:
    """숫자 키로 물체 선택. ESC → IDLE"""
    disp = ctx.grab_frame()
    cv2.imshow(WIN, draw_frame(disp, ctx.detected_snapshot, ctx.H_mat,
                               ctx.state, selected_det=ctx.current_det))
    key = read_key(1)

    if ord('1') <= key <= ord('9'):
        idx       = key - ord('1')
        non_blade = [d for d in ctx.detected_snapshot if d["cls_name"] != "Blade"]
        if idx < len(non_blade):
            ctx.current_det = non_blade[idx]
            cv2.imshow(WIN, draw_frame(disp, ctx.detected_snapshot, ctx.H_mat,
                                       ctx.state, selected_det=ctx.current_det))
            cv2.waitKey(120)
            print(f"[INPUT] [{idx+1}] {ctx.current_det['cls_name']} 선택")
            ctx.state = PLAN
        else:
            print(f"[INPUT] [{idx+1}]번 없음 (선택 가능: {len(non_blade)}개)")
    elif key == 27:
        ctx.detected_snapshot = []
        ctx.current_det = None
        ctx.state = IDLE


def state_plan(ctx: AppCtx) -> None:
    """좌표 계산 → 로봇에 execute 전송 → WAITING으로 전환"""
    target = action_plan(ctx.current_det, ctx.H_mat, ctx.detected_snapshot)
    if target is None:
        ctx.state = IDLE
        return

    approach      = action_plan_approach(target)
    blade_flipped = target.get("blade_flipped", False)
    place_joints  = list(PLACE_JOINTS_BASE)
    if target["cls_name"] == "bottle":
        place_joints[5] = math.radians(0.0)
    else:
        place_joints[5] = PLACE_J6_FLIPPED if blade_flipped else PLACE_J6_NORMAL

    ctx.rsock.send({
        "cmd":      "execute",
        "approach": approach,
        "target":   target,
        "place":    {"joints": place_joints, "cls_name": "place"},
    })
    print(f"[→WSL] execute 전송 — {target['cls_name']} "
          f"X={target['x_mm']:.0f} Y={target['y_mm']:.0f}mm")

    ctx.status_msg = f"{target['cls_name']} 작업 중..."
    ctx.state      = WAITING


def state_waiting(ctx: AppCtx) -> None:
    """로봇 작업 완료(done/failed) 수신 대기. Q → 종료"""
    disp = ctx.grab_frame()
    cv2.imshow(WIN, draw_frame(disp, ctx.detected_snapshot, ctx.H_mat,
                               ctx.state, ctx.status_msg,
                               selected_det=ctx.current_det))
    key = cv2.waitKey(30) & 0xFF
    if key in (ord('q'), 27):
        ctx.should_quit = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  핸들러 배열 — STATE_NAMES 순서와 1:1 대응
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HANDLERS = [
    state_idle,     # 0  IDLE
    state_input,    # 1  INPUT
    state_plan,     # 2  PLAN
    state_waiting,  # 3  WAITING
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  Final_Camera.py  (Windows)")
    print("=" * 60)

    model = load_model()
    H_mat = load_homography()
    cam   = open_camera()

    rsock = RobotSocket()
    if not rsock.connect(WSL_HOST, WSL_PORT):
        print("[오류] WSL 연결 실패")
        cam.release()
        sys.exit(1)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    wsl_queue: List[dict] = []
    wsl_lock = threading.Lock()

    def _recv_loop():
        while True:
            m = rsock.recv()
            if m is None:
                print("[SOCKET] WSL 연결 끊김")
                break
            with wsl_lock:
                wsl_queue.append(m)

    threading.Thread(target=_recv_loop, daemon=True).start()

    ctx = AppCtx(cam, model, H_mat, rsock, wsl_queue, wsl_lock)

    print("[START]  Space=검출  H=홈  Q=종료\n")

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    while not ctx.should_quit:
        recv_wsl(ctx)                    # WSL 메시지 처리 (상태 전환 가능)
        _HANDLERS[ctx.state](ctx)        # 현재 상태 핸들러 실행

    # ── 종료 ──────────────────────────────────────────────────────────────────
    rsock.send({"cmd": "quit"})
    cam.release()
    cv2.destroyAllWindows()
    rsock.close()
    print("[종료]")


if __name__ == "__main__":
    main()
