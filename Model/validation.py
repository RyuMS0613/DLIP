from pathlib import Path
from ultralytics import YOLO

_DIR = Path(__file__).parent

# ── 평가 대상 모델 / 데이터셋 ────────────────────────────────────
MODELS = {
    "best": (_DIR / "weights" / "best.pt", _DIR / "config" / "dataset_v2.yaml"),
}

IMGSZ  = 640
DEVICE = 0      # GPU: 0  /  CPU: 'cpu'
SPLIT  = "val"  # 'val' | 'test'


def validate_model(name: str, pt_path: Path, yaml_path: Path):
    if not pt_path.exists():
        print(f"[건너뜀] 파일 없음: {pt_path}\n")
        return None

    print("=" * 60)
    print(f"  모델  : {name}  ({pt_path.name})")
    print(f"  데이터: {yaml_path.name}  (split={SPLIT})")
    print("=" * 60)

    model = YOLO(str(pt_path))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device=DEVICE,
        split=SPLIT,
        verbose=False,
    )

    # OBB 모델: metrics.obb  /  일반 detect 모델: metrics.box
    box = metrics.obb if hasattr(metrics, "obb") and metrics.obb is not None else metrics.box

    map50    = box.map50
    map50_95 = box.map
    class_ap = box.ap_class_index   # 클래스 인덱스 배열
    ap50_per = box.ap50             # 클래스별 AP50

    names = model.names  # {0: 'Blade', 1: 'bottle', ...}

    print(f"\n  mAP50     : {map50:.4f}  ({map50*100:.2f} %)")
    print(f"  mAP50-95  : {map50_95:.4f}  ({map50_95*100:.2f} %)")
    print()
    print(f"  {'클래스':<16}  AP50")
    print(f"  {'-'*16}  ------")
    for idx, ap in zip(class_ap, ap50_per):
        cls_name = names.get(int(idx), str(idx))
        print(f"  {cls_name:<16}  {ap:.4f}")
    print()

    return {"name": name, "map50": map50, "map50_95": map50_95}


def main():
    results = []
    for name, (pt_path, yaml_path) in MODELS.items():
        r = validate_model(name, pt_path, yaml_path)
        if r:
            results.append(r)

    if len(results) > 1:
        print("=" * 60)
        print("  비교 요약")
        print("=" * 60)
        print(f"  {'모델':<16}  mAP50    mAP50-95")
        print(f"  {'-'*16}  -------  --------")
        for r in results:
            print(f"  {r['name']:<16}  {r['map50']:.4f}   {r['map50_95']:.4f}")
        print()


if __name__ == "__main__":
    main()
