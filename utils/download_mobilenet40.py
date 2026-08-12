"""
Download and extract the official modelnet40_normal_resampled dataset.

This is the correct dataset used to produce published PointNet/PointNet++
accuracy numbers (10k surface-sampled points per shape, proper mesh normals).

Usage:
    python utils/download_modelnet40.py
    python utils/download_modelnet40.py --output data/modelnet40_normal_resampled
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

URL     = "https://huggingface.co/datasets/cminst/ModelNet40/resolve/main/modelnet40_normal_resampled.zip?download=true"
OUT_DIR = "data/modelnet40_normal_resampled"


def parse_args():
    p = argparse.ArgumentParser("Download modelnet40_normal_resampled")
    p.add_argument("--output",   type=str, default=OUT_DIR,
                   help=f"Output directory (default: {OUT_DIR})")
    p.add_argument("--keep-zip", action="store_true",
                   help="Keep the zip file after extraction")
    return p.parse_args()


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done  = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  {pct:5.1f}%  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)
    else:
        print(f"\r  {downloaded / 1024 / 1024:.1f} MB downloaded", end="", flush=True)


def download_zip(zip_path: Path):
    print(f"Downloading modelnet40_normal_resampled.zip ...")
    print(f"  Source : {URL}")
    print(f"  Dest   : {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        block = 1024 * 1024  # 1 MB chunks
        block_num = 0
        while True:
            chunk = response.read(block)
            if not chunk:
                break
            f.write(chunk)
            block_num += 1
            show_progress(block_num, block, total)

    print()  # newline after progress
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  Downloaded: {size_mb:.1f} MB")


def extract_zip(zip_path: Path, output_path: Path):
    print(f"\nExtracting to {output_path.parent} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"  {len(members)} files in archive")
        for i, member in enumerate(members):
            zf.extract(member, output_path.parent)
            if i % 1000 == 0:
                print(f"\r  Extracted {i}/{len(members)} files ...", end="", flush=True)
    print(f"\r  Extracted {len(members)}/{len(members)} files.    ")

    # Rename if zip extracted to a differently-named folder
    extracted = output_path.parent / "modelnet40_normal_resampled"
    if extracted.exists() and extracted.resolve() != output_path.resolve():
        extracted.rename(output_path)
        print(f"  Moved to {output_path}")


def verify(output_path: Path):
    print("\nVerifying dataset ...")

    # Metadata files that are NOT point cloud data — exclude from counts
    METADATA_NAMES = {
        "filelist.txt", "modelnet40_shape_names.txt", "modelnet40_train.txt",
        "modelnet40_test.txt", "modelnet10_shape_names.txt",
        "modelnet10_train.txt", "modelnet10_test.txt",
    }

    all_txt   = list(output_path.rglob("*.txt"))
    # Point cloud files live inside class subdirectories (e.g. airplane/airplane_0001.txt)
    pc_files  = [f for f in all_txt if f.name not in METADATA_NAMES and f.parent.name != output_path.name]
    class_dirs = [d for d in output_path.iterdir() if d.is_dir()]

    print(f"  Classes     : {len(class_dirs)}")
    print(f"  Point cloud .txt files: {len(pc_files)}")

    if not pc_files:
        print("ERROR: No point cloud .txt files found. Extraction may have failed.")
        sys.exit(1)

    # Pick a real point cloud file (sorted → deterministic)
    sample = sorted(pc_files)[0]
    with open(sample) as f:
        lines = f.readlines()
    # Strip whitespace and split on space or comma
    first_line = lines[0].strip().replace(",", " ").split() if lines else []
    cols = len(first_line)

    print(f"  Sample      : {sample.relative_to(output_path)}")
    print(f"  Rows        : {len(lines)}  (expected ~2048–10000)")
    print(f"  Columns     : {cols}  (expected 6 — x,y,z,nx,ny,nz)")

    ok = len(lines) >= 100 and cols == 6 and len(class_dirs) == 40
    if ok:
        print("\nDataset is correct and ready for training/evaluation.")
    else:
        if len(class_dirs) != 40:
            print(f"\nWARNING: Expected 40 class directories, found {len(class_dirs)}.")
        if cols != 6:
            print(f"\nWARNING: Expected 6 columns (x,y,z,nx,ny,nz), found {cols}. Check sample file.")
        if len(lines) < 100:
            print(f"\nWARNING: Point cloud has only {len(lines)} points — may be truncated.")
        if not (len(class_dirs) == 40 and cols == 6):
            print("Run: head -2 data/modelnet40_normal_resampled/airplane/airplane_0001.txt")
            print("to inspect the actual format manually.")


def main():
    args = parse_args()
    output_path = Path(args.output)

    # Skip if already extracted
    if output_path.exists() and any(output_path.rglob("*.txt")):
        txt_count = len(list(output_path.rglob("*.txt")))
        if txt_count > 100:
            print(f"Dataset already present at {output_path} ({txt_count} .txt files).")
            verify(output_path)
            return

    zip_path = output_path.parent / "modelnet40_normal_resampled.zip"

    download_zip(zip_path)
    extract_zip(zip_path, output_path)

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"Removed zip: {zip_path}")

    verify(output_path)
    print(f"\nDone. Dataset at: {output_path.resolve()}")


if __name__ == "__main__":
    main()
