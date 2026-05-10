"""Convert an Ultralytics YOLO checkpoint to ONNX usable by fast-alpr.

If you fine-tune your own plate detector on Ultralytics (YOLO11/YOLO26/etc.),
this script exports it to the ONNX format that fast-alpr's
`LicensePlateDetector` accepts via `detector_model=<path-to-onnx>`.

Usage:

    uv run python scripts/export_ultralytics_to_onnx.py \\
        runs/detect/train/weights/best.pt \\
        models/my-plate-detector.onnx \\
        --imgsz 384 --opset 17 --simplify

Requires `ultralytics` installed (not a default dependency — add it to your
environment when you want to fine-tune or export):

    uv pip install ultralytics
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO .pt → ONNX exporter")
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint")
    parser.add_argument(
        "output", type=Path, nargs="?", help="Output .onnx path (defaults next to .pt)"
    )
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--half", action="store_true", help="FP16 export (CUDA only)")
    parser.add_argument("--device", default="cpu", help="Export device (cpu / cuda / 0)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "`ultralytics` is not installed. Run `uv pip install ultralytics` first."
        ) from e

    model = YOLO(str(args.checkpoint))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
        device=args.device,
    )
    exported_path = Path(exported)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(args.output)
        exported_path = args.output

    print(f"exported: {exported_path}")
    print(
        "Wire it in by setting ANPR_DETECTOR_MODEL to this path "
        "(absolute or relative to the working dir)."
    )


if __name__ == "__main__":
    main()
