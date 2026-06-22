#!/usr/bin/env python3
"""
YOLOv8 OBB 모델 학습

실행:
  python train_obb.py

결과:
  runs/train/weights/best.pt  -> weights/best.pt 로 복사
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

_DIR        = Path(__file__).parent
CONFIG      = _DIR / "config" / "dataset_v2.yaml"
WEIGHTS_OUT = _DIR / "weights" / "best.pt"

# ── 모델 ──────────────────────────────────────────────────────────────────
BASE_MODEL = "yolov8s-obb.pt"   # n=빠름/낮음  s=균형(권장)  m=높음/느림

# ── 기본 파라미터 ─────────────────────────────────────────────────────────
EPOCHS   = 150    # 소규모 데이터셋은 충분히 학습
IMGSZ    = 640
BATCH    = 16     # GPU 메모리 여유 있으면 32
DEVICE   = 0      # 0=GPU, 'cpu'=CPU
PATIENCE = 30     # val 개선 없으면 조기 종료 (epoch 수)

# ── Augmentation ─────────────────────────────────────────────────────────
DEGREES = 180.0   # 탑뷰 고정 — 360° 전방향 커버
SCALE   = 0.5     # 이미지 크기 ±50%
FLIPLR  = 0.5     # 좌우 반전
FLIPUD  = 0.5     # 상하 반전 (수술도구는 방향 무관)
MOSAIC  = 1.0     # mosaic augmentation

PROJECT = str(_DIR / "runs")
NAME    = "train"


def train():
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(CONFIG),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        degrees=DEGREES,
        scale=SCALE,
        fliplr=FLIPLR,
        flipud=FLIPUD,
        mosaic=MOSAIC,
        close_mosaic=20,
        cache='disk',
        workers=2,
        seed=42,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
    )
    return results


def export_best():
    src = _DIR / "runs" / NAME / "weights" / "best.pt"
    if src.exists():
        shutil.copy2(src, WEIGHTS_OUT)
        print(f"best.pt -> {WEIGHTS_OUT}")
    else:
        print(f"[경고] best.pt 를 찾을 수 없음: {src}")


if __name__ == "__main__":
    train()
    export_best()
