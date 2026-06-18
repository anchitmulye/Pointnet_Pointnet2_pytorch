"""
Download or extract ShapeNet Part Segmentation dataset.

Two modes:
  1. Auto-download from Hugging Face (requires huggingface_hub)
  2. Extract a zip you already downloaded manually (--zip_path)

Saves to: data/shapenetcore_partanno_segmentation_benchmark_v0_normal/

The HuggingFace zip has a deep internal prefix:
  home/yue/datasets/ShapeNet/shapenet_pc/shapenetcore_.../
and synset folders live inside a shape_data/ subfolder.
This script strips both so the final layout matches what ShapeNetDataLoader expects:
  data/shapenetcore_partanno_segmentation_benchmark_v0_normal/
    synsetoffset2category.txt
    train_test_split/
    02691156/   02773838/  ... (synset folders directly at root)

Manual download URL (open in browser):
  https://huggingface.co/datasets/ShapeSplats/sharing/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
"""
import io
import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path


HF_REPO_ID   = 'ShapeSplats/sharing'
HF_FILENAME  = 'shapenetcore_partanno_segmentation_benchmark_v0_normal.zip'
OUT_DIR_NAME = 'shapenetcore_partanno_segmentation_benchmark_v0_normal'

# Prefix inside the zip to strip
ZIP_PREFIX   = 'home/yue/datasets/ShapeNet/shapenet_pc/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
# Synset data lives one level deeper inside shape_data/
SHAPE_DATA   = 'shape_data/'


def _target_path(member, out_dir):
    """
    Map a zip member path to its destination on disk.

    Strips ZIP_PREFIX, then for entries under shape_data/ promotes them
    one level up so synset folders sit directly under out_dir.

    Returns None to skip an entry.
    """
    if not member.startswith(ZIP_PREFIX):
        return None                          # outside the dataset root — skip

    rel = member[len(ZIP_PREFIX):]           # strip leading prefix

    if rel == '' or rel == '/':
        return None                          # the root dir entry itself

    # Entries under shape_data/02691156/... → 02691156/...
    if rel.startswith(SHAPE_DATA):
        rel = rel[len(SHAPE_DATA):]
        if rel == '':
            return None                      # shape_data/ dir entry itself

    return out_dir / rel


def extract_zip(zip_path, data_dir):
    zip_path = Path(zip_path)
    data_dir = Path(data_dir)
    out_dir  = data_dir / OUT_DIR_NAME

    if not zip_path.is_file():
        print(f'ERROR: zip file not found: {zip_path}')
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Extracting {zip_path.name}...')
    print(f'  Destination: {out_dir}')
    print(f'  (stripping internal path prefix and shape_data/ subfolder)')
    print()

    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.infolist()
        total   = len(members)
        done    = 0

        for info in members:
            dest = _target_path(info.filename, out_dir)
            if dest is None:
                continue

            if info.filename.endswith('/'):
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(dest, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

            done += 1
            if done % 2000 == 0:
                print(f'  {done}/{total} files...', end='\r')

    print(f'  {done}/{total} files extracted.   ')
    print(f'Extraction complete. Data at: {out_dir}')
    return str(out_dir)


def download_shapenet(data_root='data', zip_path=None):
    data_dir = Path(data_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    out_dir = data_dir / OUT_DIR_NAME
    if out_dir.is_dir() and any(out_dir.iterdir()):
        print(f'Already exists: {out_dir}')
        print('Skipping.')
        return str(out_dir)

    # Mode 1: user provided a local zip (downloaded from browser)
    if zip_path:
        return extract_zip(zip_path, data_dir)

    # Mode 2: auto-download via huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('ERROR: huggingface_hub not installed.')
        print('Install with:  pip install huggingface_hub')
        print()
        print('Or download the zip manually from your browser:')
        print('  https://huggingface.co/datasets/ShapeSplats/sharing/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip')
        print()
        print('Then run:')
        print('  python utils/prepare_data_partseg.py --task extract --zip_path <path/to/downloaded.zip>')
        sys.exit(1)

    local_zip = data_dir / HF_FILENAME
    print(f'Downloading {HF_FILENAME} from Hugging Face...')
    print(f'  Repo : {HF_REPO_ID}')
    print(f'  Size : ~709 MB')
    print()

    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type='dataset',
        local_dir=str(data_dir),
    )

    result = extract_zip(local_zip, data_dir)
    os.remove(local_zip)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ShapeNet downloader / extractor')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--zip_path',  type=str, default=None,
                        help='Path to a zip already downloaded from browser')
    args = parser.parse_args()
    download_shapenet(data_root=args.data_root, zip_path=args.zip_path)