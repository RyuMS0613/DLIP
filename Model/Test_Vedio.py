#!/usr/bin/env python3
"""
Step 3: YOLO OBB 추론 단독 테스트 (로봇 없음)

출력:
  - 카메라 화면에 OBB 박스 + 중심점 + 각도 표시
  - 터미널에 픽셀 좌표, 각도 출력

실행:
  python test_inference.py

Keys:
  Q / ESC : 종료
"""

import cv2
import math
import numpy as np
from ultralytics import YOLO
import os
import sys

try:
    import PySpin
    _PYSPIN = True
except ImportError:
    _PYSPIN = False

_DIR       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_DIR, "weights", "best.pt")
CONF_THRES = 0.7

CLASS_COLORS = {
    0: (200,  80, 255),  # Blade
    1: (  0, 255,   0),  # bottle
    2: (255, 100,   0),  # scalpel10
    3: (  0, 100, 255),  # scalpel11
    4: (255,   0, 255),  # scalpel15
}
CLASS_NAMES = {0: "Blade", 1: "bottle", 2: "scalpel10", 3: "scalpel11", 4: "scalpel15"}


# ─── 카메라 래퍼 ────────────────────────────────────────────────────────────

class SpinnakerCamera:
    def __init__(self):
        self.system  = PySpin.System.GetInstance()
        cam_list     = self.system.GetCameras()
        if cam_list.GetSize() == 0:
            raise RuntimeError("Spinnaker: 카메라 없음")
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
            raise RuntimeError(f"OpenCV: 카메라 {index} 열기 실패")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


# ─── 함수 ───────────────────────────────────────────────────────────────────

def open_camera():
    """카메라 열기 (Spinnaker 우선, 없으면 OpenCV)"""
    if _PYSPIN:
        try:
            cam = SpinnakerCamera()
            print("Spinnaker 카메라 사용")
            return cam
        except Exception as e:
            print(f"Spinnaker 실패 ({e}), OpenCV로 전환")
    cam = OpenCVCamera(0)
    print("OpenCV 카메라 사용")
    return cam


def load_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[오류] 모델 없음: {MODEL_PATH}\n먼저 train_obb.py 를 실행하세요.")
    model = YOLO(MODEL_PATH)
    print(f"모델 로드: {MODEL_PATH}")
    return model


def parse_detections(results) -> list:
    """
    반환: [{"cls": int, "cx": px, "cy": py, "angle_deg": deg, "conf": float}, ...]
    angle_deg : YOLO OBB xywhr 의 r(rad) → degree 변환값
    """
    detections = []
    for result in results:
        if result.obb is None:
            continue
        xywhr = result.obb.xywhr.cpu().numpy()   # (N, 5): cx cy w h r
        confs  = result.obb.conf.cpu().numpy()
        clss   = result.obb.cls.cpu().numpy().astype(int)
        for (cx, cy, w, h, r), conf, cls in zip(xywhr, confs, clss):
            detections.append({
                "cls":       int(cls),
                "cx":        float(cx),
                "cy":        float(cy),
                "angle_deg": float(math.degrees(r)),
                "conf":      float(conf),
            })
    return detections


def draw_obb(frame: np.ndarray, results, snapshot: bool = False,
             selected_idx: int = None) -> np.ndarray:
    """OBB 박스, 중심점, 각도를 이미지에 그리기.
    snapshot=True 이면 Blade 외 물체를 번호 붙여 표시.
    selected_idx는 Blade 제외 물체 기준 0부터 시작하는 선택 인덱스."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    non_blade_idx = 0

    for result in results:
        if result.obb is None:
            continue
        corners_all = result.obb.xyxyxyxy.cpu().numpy().astype(np.int32)
        xywhr       = result.obb.xywhr.cpu().numpy()
        confs       = result.obb.conf.cpu().numpy()
        clss        = result.obb.cls.cpu().numpy().astype(int)

        for corners, (cx, cy, *_, r), conf, cls in zip(
                corners_all, xywhr, confs, clss):
            name = CLASS_NAMES.get(int(cls), str(cls))
            angle_deg = math.degrees(r)

            if snapshot and name != "Blade":
                is_selected = (selected_idx == non_blade_idx)
                color     = (0, 0, 255) if is_selected else (0, 220, 220)
                thickness = 3 if is_selected else 2
                label = f"[{non_blade_idx + 1}] {name} {conf:.2f} | {angle_deg:.1f}deg"
                non_blade_idx += 1
            elif snapshot and name == "Blade":
                color     = (100, 100, 200)
                thickness = 1
                label = f"Blade {conf:.2f}"
            else:
                color     = CLASS_COLORS.get(int(cls), (200, 200, 200))
                thickness = 2
                label = f"{name} {conf:.2f} | {angle_deg:.1f}deg"

            cv2.polylines(out, [corners], isClosed=True, color=color, thickness=thickness)
            cx_i, cy_i = int(cx), int(cy)
            cv2.circle(out, (cx_i, cy_i), 5, color, -1)
            cv2.putText(out, label, (cx_i - 60, cy_i - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 하단 상태 바
    bar_color = (0, 0, 200) if snapshot else (0, 160, 0)
    bar_color = (0, 0, 200) if snapshot else (0, 160, 0)
    bar_text  = "SNAPSHOT  —  1~9: 선택  |  ESC: 라이브  |  Q: 종료" if snapshot else \
                "LIVE  —  Space: 스냅샷  |  Q/ESC: 종료"
    cv2.rectangle(out, (0, fh - 26), (fw, fh), (20, 20, 20), -1)
    cv2.putText(out, bar_text, (8, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_color, 1, cv2.LINE_AA)
    return out


def infer_image(image_path: str, model: YOLO = None, show: bool = True) -> list:
    """
    이미지 파일 한 장에 대해 YOLO OBB 추론.

    Args:
        image_path : 이미지 파일 경로
        model      : 이미 로드된 YOLO 모델 (None이면 내부에서 로드)
        show       : True → 결과 창 표시 (아무 키 누르면 닫힘)

    Returns:
        parse_detections()와 동일한 리스트
        [{"cls", "cx", "cy", "angle_deg", "conf"}, ...]
    """
    if model is None:
        model = load_model()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[오류] 이미지 로드 실패: {image_path}")
        return []

    results    = model(frame, conf=CONF_THRES, verbose=False)
    detections = parse_detections(results)

    print(f"\n[infer_image] {os.path.basename(image_path)}  —  {len(detections)}개 검출")
    for d in detections:
        print(f"  [{CLASS_NAMES.get(d['cls'], d['cls'])}]  "
              f"cx={d['cx']:.0f}  cy={d['cy']:.0f}  "
              f"angle={d['angle_deg']:.1f}°  conf={d['conf']:.2f}")

    if show:
        cv2.imshow("infer_image", draw_obb(frame, results))
        cv2.waitKey(0)
        cv2.destroyWindow("infer_image")

    return detections


def main():
    model    = load_model()
    cam      = open_camera()
    snapshot = False   # True: 스냅샷 선택 모드
    snap_frame = None
    snap_results = None
    selected_idx = None

    print("추론 시작 — Space: 스냅샷  |  Q / ESC: 종료")
    while True:
        if not snapshot:
            frame = cam.read()
            if frame is None:
                continue
            results    = model(frame, conf=CONF_THRES, verbose=False)
            detections = parse_detections(results)
            cv2.imshow("OBB Inference", draw_obb(frame, results, snapshot=False))
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # Space
                snap_frame   = frame.copy()
                snap_results = results
                selected_idx = None
                snapshot    = True
                non_blade = [d for d in detections if CLASS_NAMES.get(d["cls"]) != "Blade"]
                print(f"\n[SNAPSHOT] {len(non_blade)}개 검출:")
                for i, d in enumerate(non_blade):
                    print(f"  [{i+1}] {CLASS_NAMES.get(d['cls'], d['cls']):<12} "
                          f"cx={d['cx']:.0f}  cy={d['cy']:.0f}  "
                          f"angle={d['angle_deg']:.1f}°  conf={d['conf']:.2f}")
                cv2.imshow("OBB Inference", draw_obb(snap_frame, snap_results, snapshot=True))

            elif key in (ord('q'), 27):
                break

        else:  # 스냅샷 선택 모드
            cv2.imshow("OBB Inference",
                       draw_obb(snap_frame, snap_results, snapshot=True,
                                selected_idx=selected_idx))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key == 27:
                snapshot = False
                snap_frame = None
                snap_results = None
                selected_idx = None
                print("[LIVE] restart\n")
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                detections = parse_detections(snap_results)
                non_blade = [d for d in detections if CLASS_NAMES.get(d["cls"]) != "Blade"]
                if idx < len(non_blade):
                    selected_idx = idx
                    name = CLASS_NAMES.get(non_blade[idx]["cls"], non_blade[idx]["cls"])
                    print(f"[SELECT] [{idx+1}] {name}")
                else:
                    print(f"[SELECT] [{idx+1}] not found (available: {len(non_blade)})")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 카메라 라이브 추론
    main()

    # 이미지 파일 추론 예시 (주석 해제해서 사용)
    # infer_image("test.jpg")
