import argparse
import glob
import inspect
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pydicom
from pydicom.errors import InvalidDicomError


def is_series_dir(path: str) -> bool:
    """Heuristic: a series dir contains many DICOM/IMA files (>=10)."""
    if not os.path.isdir(path):
        return False
    files = [f for f in glob.glob(os.path.join(path, "*")) if os.path.isfile(f)]
    if len(files) < 10:
        return False
    # quick extension check (IMA/DICOM, but some files may lack extension)
    sample = files[0]
    ext = os.path.splitext(sample)[1].lower()
    return True if ext in (".ima", ".dcm", "") else True


def has_b30_kernel(series_dir: str) -> bool:
    """Return True if series ConvolutionKernel contains 'B30' (case-insensitive)."""
    # scan a few files to be safe
    for f in sorted(glob.glob(os.path.join(series_dir, "*")))[:5]:
        if not os.path.isfile(f):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except (InvalidDicomError, Exception):
            continue
        kernel = getattr(ds, "ConvolutionKernel", "")
        if isinstance(kernel, (list, tuple)):
            kernel = " ".join(kernel)
        if "B30" in str(kernel).upper():
            return True
    return False


def find_series(data_roots, dose: Literal["quarter", "full"] = "quarter"):
    """
    Discover patient series under the given data_roots.
    We look for .../Lx/{dose}_1mm directories that contain DICOM/IMA slices.
    Returns list of (root_name, Lx, leaf_name, series_dir)
    """
    found = []
    print(f"Searching for '{dose}_1mm' series under: {data_roots}")
    for root in data_roots:
        root = os.path.abspath(root)
        root_name = os.path.basename(root.rstrip("/"))
        for Ldir in sorted(glob.glob(os.path.join(root, "L*"))):
            if not os.path.isdir(Ldir):
                continue
            Lname = os.path.basename(Ldir)
            leaf = f"{dose}_1mm"
            series_dir = os.path.join(Ldir, leaf)
            if is_series_dir(series_dir):
                found.append((root_name, Lname, leaf, series_dir))
    return found


def out_path_for(out_root, root_name, Lname, leaf):
    return os.path.join(out_root, root_name, Lname, leaf)


def _device_preference():
    # Prefer CUDA, then MPS (macOS), then CPU
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_totalseg(input_dir, output_dir, device="mps", overwrite=False):
    """
    Run TotalSegmentator python API on a single series directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    seg_path = os.path.join(output_dir, "segmentations.nii.gz")

    if os.path.exists(seg_path) and not overwrite:
        return f"SKIP (exists): {seg_path}"

    # Sanity: ensure it's CT with B30 kernel
    if not has_b30_kernel(input_dir):
        return f"SKIP (non-B30 kernel): {input_dir}"

    # Import here (inside worker) to avoid conflicts with torch initialization
    from totalsegmentator import python_api

    # Build kwargs dynamically to match API signature (device vs device_type naming)
    sig = inspect.signature(python_api.totalsegmentator)
    kwargs = {
        "input": input_dir,
        "output": output_dir,
        "ml": True,  # multilabel
        "task": "tissue_4_types",  # 4 tissue classes
        "fast": False,  # important: fast not supported for this task
        "preview": False,
    }
    if "device" in sig.parameters:
        kwargs["device"] = device
    elif "device_type" in sig.parameters:
        kwargs["device_type"] = device

    # (Optional) control threads (safer on laptops)
    if "nr_thr_resamp" in sig.parameters:
        kwargs["nr_thr_resamp"] = 1
    if "nr_thr_saving" in sig.parameters:
        kwargs["nr_thr_saving"] = 1

    # Run + simple log
    log_path = os.path.join(output_dir, "log.txt")
    try:
        res = python_api.totalsegmentator(
            **kwargs,
        )
        with open(log_path, "w") as fp:
            fp.write(f"OK {input_dir}\nDevice={device}\nArgs={kwargs}\nResult={res}\n")
        return f"OK: {seg_path}"
    except Exception as e:
        with open(log_path, "w") as fp:
            fp.write("ERROR\n")
            fp.write(str(e) + "\n")
            fp.write(traceback.format_exc())
        return f"ERROR: {input_dir} -> {e}"


def main():
    ap = argparse.ArgumentParser(
        description="Batch TotalSegmentator (tissue_4_types) over Mayo QD 1mm B30 series."
    )
    ap.add_argument(
        "--data_roots",
        nargs="+",
        required=True,
        help="Paths to roots that contain Lx folders with quarter_1mm DICOM series.",
    )
    ap.add_argument(
        "--dose",
        choices=["quarter", "full"],
        default="quarter",
        help="Dose type to look for (default: quarter, as used in the paper).",
    )
    ap.add_argument(
        "--out_root",
        default="segmentations",
        help="Root for mirrored outputs (default: segmentations)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers. Default: 1 for GPU devices, min(4, cpu_count) for CPU.",
    )
    ap.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device for TotalSegmentator. 'auto' picks cuda > mps > cpu.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if segmentations.nii.gz exists.",
    )
    args = ap.parse_args()

    device = _device_preference() if args.device == "auto" else args.device

    series = find_series(args.data_roots, args.dose)
    if not series:
        print(
            "No series found. Check --data_roots (expecting Lx/quarter_1mm subdirs).",
            file=sys.stderr,
        )
        sys.exit(2)

    # Build jobs list (with B30 filter check deferred to worker)
    jobs = []
    for root_name, Lname, leaf, series_dir in series:
        out_dir = out_path_for(args.out_root, root_name, Lname, leaf)
        jobs.append((series_dir, out_dir))

    # Pick sensible default workers: GPU runs one series at a time; CPU can parallelise
    if args.workers is None:
        import multiprocessing as mp

        cpu = max(1, mp.cpu_count())
        args.workers = 1 if device in ("cuda", "mps") else min(4, cpu)

    print(f"Found {len(jobs)} series.")
    print(f"Device: {device} | Workers (threads): {args.workers}")
    print(f"Output root: {os.path.abspath(args.out_root)}")
    os.makedirs(args.out_root, exist_ok=True)

    # Run in parallel (threads)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_totalseg, inp, out, device, args.overwrite): (inp, out)
            for (inp, out) in jobs
        }
        for fut in as_completed(futs):
            inp, out = futs[fut]
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"ERROR (crashed): {inp} -> {e}"
            print(msg)
            results.append(msg)

    # Summary
    ok = sum(1 for m in results if m.startswith("OK"))
    skip = sum(1 for m in results if m.startswith("SKIP"))
    err = sum(1 for m in results if m.startswith("ERROR"))
    print(f"\nDone. OK={ok}  SKIP={skip}  ERROR={err}")


if __name__ == "__main__":
    main()
