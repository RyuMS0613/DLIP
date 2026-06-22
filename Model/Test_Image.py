"""
임의 이미지로 YOLO OBB 추론 테스트

사용법:
  python infer_image.py <이미지경로>          # 단일 이미지
  python infer_image.py <이미지경로> --save   # 결과 이미지 저장 (output/ 폴더)
  python infer_image.py                       # 인자 없으면 파일 선택 다이얼로그

Keys (창이 열렸을 때):
  S     : 현재 결과 이미지 저장
  Q/ESC : 종료
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

_DIR       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_DIR, "weights", "best_blades.pt")
CONF_THRES = 0.3

CLASS_COLORS = {
    0: (0, 200, 255),
    1: (0, 255, 0),
    2: (255, 100, 0),
    3: (0, 100, 255),
    4: (255, 0, 255),
}
CLASS_NAMES = {0: "Blade", 1: "bottle", 2: "scalpel10", 3: "scalpel11", 4: "scalpel15"}


def load_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[오류] 모델 없음: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"모델 로드: {MODEL_PATH}")
    return model


def draw_obb(frame: np.ndarray, results) -> np.ndarray:
    out = frame.copy()
    for result in results:
        if result.obb is None:
            continue
        corners_all = result.obb.xyxyxyxy.cpu().numpy().astype(np.int32)
        xywhr       = result.obb.xywhr.cpu().numpy()
        confs       = result.obb.conf.cpu().numpy()
        clss        = result.obb.cls.cpu().numpy().astype(int)

        for corners, (cx, cy, *_, r), conf, cls in zip(corners_all, xywhr, confs, clss):
            color = CLASS_COLORS.get(int(cls), (200, 200, 200))
            name  = CLASS_NAMES.get(int(cls), str(cls))

            cv2.polylines(out, [corners], isClosed=True, color=color, thickness=2)
            cx_i, cy_i = int(cx), int(cy)
            cv2.circle(out, (cx_i, cy_i), 6, color, -1)

            angle_deg = math.degrees(r)
            label = f"{name} {conf:.2f} | {angle_deg:.1f}deg"
            cv2.putText(out, label, (cx_i - 60, cy_i - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def print_detections(results):
    found = False
    for result in results:
        if result.obb is None:
            continue
        xywhr = result.obb.xywhr.cpu().numpy()
        confs  = result.obb.conf.cpu().numpy()
        clss   = result.obb.cls.cpu().numpy().astype(int)
        for (cx, cy, w, h, r), conf, cls in zip(xywhr, confs, clss):
            name = CLASS_NAMES.get(int(cls), str(cls))
            print(f"  [{name}]  cx={cx:.0f}  cy={cy:.0f}  "
                  f"w={w:.0f}  h={h:.0f}  angle={math.degrees(r):.1f}°  conf={conf:.2f}")
            found = True
    if not found:
        print("  (탐지 없음)")


def save_result(out_img: np.ndarray, src_path: str):
    out_dir = os.path.join(_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, f"{basename}_result.jpg")
    cv2.imwrite(out_path, out_img)
    print(f"저장: {out_path}")


def pick_file() -> str:
    """tkinter 파일 선택 다이얼로그"""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="이미지 선택",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")]
    )
    root.destroy()
    return path


def main():
    parser = argparse.ArgumentParser(description="YOLO OBB 이미지 추론")
    parser.add_argument("image", nargs="?", help="이미지 파일 경로")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help="신뢰도 임계값")
    parser.add_argument("--save", action="store_true", help="결과 이미지 자동 저장")
    args = parser.parse_args()

    img_path = args.image
    if not img_path:
        img_path = pick_file()
    if not img_path:
        sys.exit("이미지 경로가 없습니다.")
    if not os.path.exists(img_path):
        sys.exit(f"[오류] 파일 없음: {img_path}")

    frame = cv2.imread(img_path)
    if frame is None:
        sys.exit(f"[오류] 이미지 읽기 실패: {img_path}")

    model   = load_model()
    results = model(frame, conf=args.conf, verbose=False)

    print(f"\n[결과] {os.path.basename(img_path)}")
    print_detections(results)

    out_img = draw_obb(frame, results)

    if args.save:
        save_result(out_img, img_path)

    # 화면 표시
    win_title = f"OBB — {os.path.basename(img_path)}"
    cv2.imshow(win_title, out_img)
    print("\nS: 저장  |  Q/ESC: 종료")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            save_result(out_img, img_path)
        elif key in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
