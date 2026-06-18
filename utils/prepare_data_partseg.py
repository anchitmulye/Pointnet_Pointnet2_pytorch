"""
Data Preparation Script for Part Segmentation (ShapeNet)
Mirrors prepare_data.py used for classification.

Three ways to get the data:

  1. Auto-download from Hugging Face (requires huggingface_hub + network access):
       python utils/prepare_data_partseg.py --task download

  2. Extract a zip you downloaded manually from the browser:
       python utils/prepare_data_partseg.py --task extract --zip_path <path/to/file.zip>

     Browser download URL:
       https://huggingface.co/datasets/ShapeSplats/sharing/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

  3. Verify dataset structure after extraction:
       python utils/prepare_data_partseg.py --task check

  4. Download + verify in one step:
       python utils/prepare_data_partseg.py --task all
"""

import os
import sys
import argparse

sys.path.append('utils')


def download_shapenet(data_root='data', zip_path=None):
    from download_shapenet import download_shapenet as _download
    print('\nDownloading ShapeNet Part Segmentation dataset...')
    out_dir = _download(data_root=data_root, zip_path=zip_path)
    print('Done!')
    return out_dir


def check_shapenet(data_dir='data/shapenetcore_partanno_segmentation_benchmark_v0_normal'):
    from verify_shapenet import verify_shapenet as _verify
    print('\nChecking ShapeNet dataset...')
    return _verify(data_dir=data_dir)


def main():
    parser = argparse.ArgumentParser('ShapeNet Part Segmentation Data Preparation')
    parser.add_argument('--task', type=str, required=True,
                        choices=['download', 'extract', 'check', 'all'],
                        help=(
                            'download : auto-download from Hugging Face  |  '
                            'extract  : extract a zip downloaded from browser (use --zip_path)  |  '
                            'check    : verify dataset structure  |  '
                            'all      : download then check'
                        ))
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root data directory (default: data/)')
    parser.add_argument('--zip_path', type=str, default=None,
                        help='Path to zip file already downloaded from browser (used with --task extract)')
    args = parser.parse_args()

    data_dir = os.path.join(
        args.data_root,
        'shapenetcore_partanno_segmentation_benchmark_v0_normal'
    )

    if args.task == 'download' or args.task == 'all':
        download_shapenet(data_root=args.data_root)

    elif args.task == 'extract':
        if not args.zip_path:
            print('ERROR: --zip_path is required for --task extract')
            print()
            print('Example:')
            print('  python utils/prepare_data_partseg.py --task extract --zip_path C:/Users/you/Downloads/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip')
            return
        download_shapenet(data_root=args.data_root, zip_path=args.zip_path)

    if args.task in ('check', 'all', 'extract'):
        check_shapenet(data_dir=data_dir)

    print('\n' + '=' * 60)
    print('Data preparation complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()