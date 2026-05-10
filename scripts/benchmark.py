"""Latency benchmark for the inference pipeline.

Loads fast-alpr (or any other configured detector + reader) and runs N
iterations against a sample image, reporting per-stage timing. Useful for
comparing CPU vs CUDA vs OpenVINO inference providers when ANPR_DEVICE
is set, or measuring the delta of a fine-tuned model.

Usage:

    uv run python scripts/benchmark.py path/to/image.jpg --iterations 50
    ANPR_DEVICE=openvino uv run python scripts/benchmark.py img.jpg
    ANPR_DEVICE=cuda uv run python scripts/benchmark.py img.jpg

Prints median, p95, p99 latencies for: total, detection, OCR.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2

from anpr.config import get_settings
from anpr.detector.fast_alpr import FastAlprDetector
from anpr.ocr.fast_plate import FastPlateOcr
from anpr.postprocess.turkish import parse_turkish_plate


def _q(samples: list[float], q: float) -> float:
    return statistics.quantiles(samples, n=100)[int(q * 100) - 1] if samples else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="anpr inference latency benchmark")
    parser.add_argument("image", type=Path, help="Path to a sample image")
    parser.add_argument("--iterations", "-n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"could not decode image: {args.image}")

    settings = get_settings()
    print(f"detector: {settings.detector_model}")
    print(f"ocr:      {settings.ocr_model}")
    print(f"device:   {settings.device}")
    print(f"image:    {args.image} ({image.shape[1]}x{image.shape[0]})")
    print()

    detector = FastAlprDetector(
        model_name=settings.detector_model,
        conf_threshold=settings.detector_conf_threshold,
    )
    reader = FastPlateOcr(model_name=settings.ocr_model, device=settings.device)

    detection_times: list[float] = []
    ocr_times: list[float] = []
    total_times: list[float] = []
    plate_count = 0
    parsed_count = 0

    for i in range(args.warmup + args.iterations):
        t0 = time.perf_counter()
        detections = detector.detect(image)
        t1 = time.perf_counter()

        crops = [d.bbox.crop(image) for d in detections]
        results = reader.read_batch(crops) if crops else []
        t2 = time.perf_counter()

        if i >= args.warmup:
            detection_times.append((t1 - t0) * 1000)
            ocr_times.append((t2 - t1) * 1000)
            total_times.append((t2 - t0) * 1000)
            plate_count += len(detections)
            for r in results:
                if r and parse_turkish_plate(r.text) is not None:
                    parsed_count += 1

    def summary(label: str, samples: list[float]) -> None:
        if not samples:
            print(f"{label:>14}: no samples")
            return
        print(
            f"{label:>14}:  median {statistics.median(samples):7.2f} ms  "
            f"p95 {_q(samples, 0.95):7.2f} ms  "
            f"p99 {_q(samples, 0.99):7.2f} ms  "
            f"min {min(samples):7.2f}  max {max(samples):7.2f}"
        )

    print(f"iterations: {args.iterations}  warmup: {args.warmup}")
    print(f"detections: {plate_count}  parsed TR: {parsed_count}")
    print()
    summary("detect", detection_times)
    summary("ocr", ocr_times)
    summary("total", total_times)


if __name__ == "__main__":
    main()
