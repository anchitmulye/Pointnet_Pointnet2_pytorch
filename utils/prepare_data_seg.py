"""
Data Preparation Script for Segmentation Tasks
Mirrors prepare_data.py for classification, covering:
  - ShapeNet (part segmentation)
  - S3DIS / Stanford3dDataset (semantic segmentation)

Usage:
    # Part segmentation — download ShapeNet
    python utils/prepare_data_seg.py --task download --seg_task part_seg

    # Semantic segmentation — verify S3DIS raw data is in place
    python utils/prepare_data_seg.py --task check --seg_task sem_seg

    # Semantic segmentation — convert raw S3DIS rooms to .npy files
    python utils/prepare_data_seg.py --task convert --seg_task sem_seg

    # Semantic segmentation — convert specific area only
    python utils/prepare_data_seg.py --task convert --seg_task sem_seg --area 5

    # Run check + convert in one shot
    python utils/prepare_data_seg.py --task all --seg_task sem_seg
"""

import argparse
import glob
import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'data_utils'))


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SHAPENET_DIR  = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')

S3DIS_RAW_DIR = os.path.join(ROOT_DIR, 'data', 's3dis', 'Stanford3dDataset_v1.2_Aligned_Version')
S3DIS_OUT_DIR = os.path.join(ROOT_DIR, 'data', 's3dis', 'stanford_indoor3d')

ANNO_PATHS_FILE = os.path.join(ROOT_DIR, 'data_utils', 'meta', 'anno_paths.txt')


# ─────────────────────────────────────────────────────────────────────────────
# Part segmentation — ShapeNet
# ─────────────────────────────────────────────────────────────────────────────

def download_shapenet():
    """Download ShapeNet part-seg dataset from Hugging Face."""
    from download_shapenet import download_shapenet as _download
    _download(data_root=os.path.join(ROOT_DIR, 'data'))


def check_shapenet():
    """Verify ShapeNet directory structure."""
    from verify_shapenet import verify_shapenet as _verify
    return _verify(data_dir=SHAPENET_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic segmentation — S3DIS
# ─────────────────────────────────────────────────────────────────────────────

def check_s3dis_raw(area=None):
    """Verify the raw Stanford3dDataset is present and structurally sound."""
    print('\n── S3DIS raw data check ────────────────────────────────')

    if not os.path.isdir(S3DIS_RAW_DIR):
        print(f'[MISSING] {S3DIS_RAW_DIR}')
        print()
        print('  S3DIS must be downloaded manually from Stanford:')
        print('  1. Request access at: http://buildingparser.stanford.edu/dataset.html')
        print('  2. Download:  Stanford3dDataset_v1.2_Aligned_Version.zip')
        print('  3. Extract to:')
        print(f'       {os.path.join(ROOT_DIR, "data", "s3dis")}')
        print('  4. Final structure should be:')
        print(f'       {S3DIS_RAW_DIR}/')
        print('         Area_1/  Area_2/  ...  Area_6/')
        print()
        return False

    areas = sorted(d for d in os.listdir(S3DIS_RAW_DIR) if d.startswith('Area_'))
    if not areas:
        print(f'[ERROR] No Area_* directories found in {S3DIS_RAW_DIR}')
        return False

    area_filter = [f'Area_{area}'] if area else areas
    total_rooms = total_missing = 0

    for a in area_filter:
        area_path = os.path.join(S3DIS_RAW_DIR, a)
        if not os.path.isdir(area_path):
            print(f'  [MISSING] {a}')
            continue
        rooms = [r for r in os.listdir(area_path)
                 if os.path.isdir(os.path.join(area_path, r))]
        missing_anno = []
        for room in rooms:
            anno = os.path.join(area_path, room, 'Annotations')
            if not os.path.isdir(anno):
                missing_anno.append(room)
        total_rooms += len(rooms)
        total_missing += len(missing_anno)
        status = f'{len(rooms)} rooms' + (f', {len(missing_anno)} missing Annotations' if missing_anno else ', all OK')
        print(f'  {a}: {status}')

    print(f'\n  Total rooms: {total_rooms}, missing Annotations: {total_missing}')

    if total_missing > 0:
        print('\n  [WARNING] Some rooms are missing Annotations folders.')
        print('  Conversion will skip those rooms.')
    else:
        print('  Raw data looks complete.')

    return total_rooms > 0


def convert_s3dis(area=None):
    """
    Convert raw S3DIS room Annotations into per-room .npy files.
    Each output file: Area_N_roomname.npy  shape (num_points, 7)  columns: X Y Z R G B label
    Output directory: data/s3dis/stanford_indoor3d/
    """
    print('\n── S3DIS conversion ────────────────────────────────────')

    if not os.path.isdir(S3DIS_RAW_DIR):
        print(f'[ERROR] Raw data not found: {S3DIS_RAW_DIR}')
        print('Run --task check first for instructions.')
        return

    from indoor3d_util import g_classes, g_class2label

    os.makedirs(S3DIS_OUT_DIR, exist_ok=True)
    print(f'Output dir: {S3DIS_OUT_DIR}')

    # Collect annotation paths from meta file, or discover from raw dir
    anno_paths = _collect_anno_paths(area)
    if not anno_paths:
        print('[ERROR] No annotation paths found. Check the raw data directory.')
        return

    print(f'Processing {len(anno_paths)} rooms...\n')

    ok = skipped = errors = 0
    for anno_path in anno_paths:
        # Build output filename: Area_N_roomname.npy
        # anno_path looks like: .../Stanford3dDataset.../Area_1/office_2/Annotations
        parts = anno_path.replace('\\', '/').split('/')
        try:
            area_part = next(p for p in parts if p.startswith('Area_'))
            room_part = parts[parts.index(area_part) + 1]
        except (StopIteration, ValueError):
            print(f'  [SKIP] Cannot parse path: {anno_path}')
            skipped += 1
            continue

        out_filename = f'{area_part}_{room_part}.npy'
        out_path = os.path.join(S3DIS_OUT_DIR, out_filename)

        if os.path.isfile(out_path):
            print(f'  [EXISTS] {out_filename}')
            skipped += 1
            continue

        if not os.path.isdir(anno_path):
            print(f'  [MISSING] {anno_path}')
            skipped += 1
            continue

        try:
            _collect_point_label(anno_path, out_path, g_classes, g_class2label)
            print(f'  [OK] {out_filename}')
            ok += 1
        except Exception as e:
            print(f'  [ERROR] {out_filename}: {e}')
            errors += 1

    print(f'\nDone.  converted: {ok}  skipped/exists: {skipped}  errors: {errors}')
    print(f'Output: {S3DIS_OUT_DIR}')


def check_s3dis_converted(area=None):
    """Verify the converted .npy files are present."""
    print('\n── S3DIS converted data check ──────────────────────────')

    if not os.path.isdir(S3DIS_OUT_DIR):
        print(f'[MISSING] {S3DIS_OUT_DIR}')
        print('  Run:  python utils/prepare_data_seg.py --task convert --seg_task sem_seg')
        return False

    npy_files = sorted(glob.glob(os.path.join(S3DIS_OUT_DIR, '*.npy')))
    if not npy_files:
        print(f'No .npy files found in {S3DIS_OUT_DIR}')
        return False

    # Group by area
    from collections import defaultdict
    by_area = defaultdict(list)
    for f in npy_files:
        name = os.path.basename(f)
        area_tag = name.split('_')[0] + '_' + name.split('_')[1]  # Area_N
        by_area[area_tag].append(name)

    area_filter = [f'Area_{area}'] if area else None
    total = 0
    for a in sorted(by_area):
        if area_filter and a not in area_filter:
            continue
        count = len(by_area[a])
        total += count
        print(f'  {a}: {count} rooms')

    print(f'\n  Total .npy files: {total}')
    print(f'  Expected: ~272 rooms across 6 areas (varies by version)')

    # Quick sanity-check on one file
    import numpy as np
    sample = npy_files[0]
    data = np.load(sample)
    print(f'\n  Sample file: {os.path.basename(sample)}')
    print(f'    Shape: {data.shape}  (expected: N x 7)')
    print(f'    Columns: X Y Z R G B label')
    print(f'    Labels present: {sorted(set(data[:, 6].astype(int).tolist()))}')

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_anno_paths(area=None):
    """Build list of Annotation directory paths to process."""
    if area:
        pattern = os.path.join(S3DIS_RAW_DIR, f'Area_{area}', '*', 'Annotations')
        return sorted(glob.glob(pattern))

    # Use meta file if available, else discover from filesystem
    if os.path.isfile(ANNO_PATHS_FILE):
        with open(ANNO_PATHS_FILE) as f:
            rel_paths = [line.strip() for line in f if line.strip()]
        return [os.path.join(S3DIS_RAW_DIR, p) for p in rel_paths]

    return sorted(glob.glob(os.path.join(S3DIS_RAW_DIR, 'Area_*', '*', 'Annotations')))


def _collect_point_label(anno_path, out_path, g_classes, g_class2label):
    """
    Read all per-object .txt files in anno_path, merge into a single
    (N, 7) array [X Y Z R G B label] and save as .npy.
    Mirrors indoor3d_util.collect_point_label with numpy output.
    """
    import numpy as np

    points_list = []
    for txt_file in sorted(glob.glob(os.path.join(anno_path, '*.txt'))):
        cls = os.path.basename(txt_file).split('_')[0]
        if cls not in g_classes:
            cls = 'clutter'
        try:
            pts = np.loadtxt(txt_file)
        except Exception as e:
            # Some files have stray characters — try with a lenient load
            pts = np.genfromtxt(txt_file, invalid_raise=False)
            pts = pts[~np.isnan(pts).any(axis=1)]
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        label = np.full((pts.shape[0], 1), g_class2label[cls], dtype=np.float32)
        points_list.append(np.concatenate([pts[:, :6], label], axis=1))

    data = np.concatenate(points_list, axis=0)
    # Shift so minimum XYZ is at origin
    data[:, :3] -= np.amin(data[:, :3], axis=0)
    np.save(out_path, data.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('Segmentation Data Preparation')
    p.add_argument('--task', required=True,
                   choices=['download', 'check', 'convert', 'all'],
                   help=(
                       'download: fetch dataset  |  '
                       'check: verify files are in place  |  '
                       'convert: process raw S3DIS rooms to .npy  |  '
                       'all: check + convert'
                   ))
    p.add_argument('--seg_task', required=True,
                   choices=['part_seg', 'sem_seg'],
                   help='part_seg = ShapeNet  |  sem_seg = S3DIS')
    p.add_argument('--area', type=int, default=None, choices=[1,2,3,4,5,6],
                   help='S3DIS only: process/check a single area (1–6). Default: all areas.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.seg_task == 'part_seg':
        if args.task in ('download', 'all'):
            download_shapenet()
        if args.task in ('check', 'all'):
            check_shapenet()
        if args.task == 'convert':
            print('ShapeNet requires no conversion — the loader reads raw .txt files directly.')

    elif args.seg_task == 'sem_seg':
        if args.task == 'download':
            print('\nS3DIS cannot be downloaded automatically.')
            print('It requires a manual access request from Stanford University.')
            print()
            print('Steps:')
            print('  1. Visit:   http://buildingparser.stanford.edu/dataset.html')
            print('  2. Request access and download:')
            print('       Stanford3dDataset_v1.2_Aligned_Version.zip')
            print('  3. Extract into:')
            print(f'       {os.path.join(ROOT_DIR, "data", "s3dis")}')
            print('  4. Expected final path:')
            print(f'       {S3DIS_RAW_DIR}')
            print('       └── Area_1/  Area_2/  ...  Area_6/')
            print()
            print('  Then run:')
            print('       python utils/prepare_data_seg.py --task convert --seg_task sem_seg')

        elif args.task == 'check':
            ok = check_s3dis_raw(args.area)
            if ok and os.path.isdir(S3DIS_OUT_DIR):
                check_s3dis_converted(args.area)

        elif args.task == 'convert':
            if check_s3dis_raw(args.area):
                convert_s3dis(args.area)
            else:
                print('\nFix raw data issues above before converting.')

        elif args.task == 'all':
            if check_s3dis_raw(args.area):
                convert_s3dis(args.area)
                check_s3dis_converted(args.area)

    print('\n' + '=' * 60)
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    main()