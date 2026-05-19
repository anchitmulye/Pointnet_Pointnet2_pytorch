"""
Data Preparation Script
Handles all data preparation tasks: download, convert, verify
"""

import os
import sys
import argparse
from pathlib import Path

# Add utils to path
sys.path.append('utils')

# def check_data_structure(data_dir='data/ModelNet40'):
#     """Verify data structure is correct"""
#     from check_data_structure import check_structure
#     print(f"\nChecking data structure in: {data_dir}")
#     check_structure(data_dir)
#     print("✓ Data structure verified!")

def convert_modelnet(source_dir, target_dir='data/modelnet40_normal_resampled', num_points=1024):
    """Convert ModelNet data to required format"""
    from convert_data import convert_modelnet
    print(f"\nConverting ModelNet data...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    convert_modelnet(source_dir, target_dir, num_points)
    print("✓ Conversion complete!")

def generate_metadata(data_dir='data/ModelNet40'):
    """Generate metadata for dataset"""
    from generate_metadata import generate_metadata
    print(f"\nGenerating metadata for: {data_dir}")
    generate_metadata(data_dir)
    print("✓ Metadata generated!")

def download_modelnet():
    """Download ModelNet40 dataset"""
    print("\nDownloading ModelNet40 dataset...")
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    target = data_dir / "modelnet40_normal_resampled.zip"

    import urllib.request
    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, target)

    print("Extracting...")
    import zipfile
    with zipfile.ZipFile(target, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(target)
    print("Download complete!")

def main():
    parser = argparse.ArgumentParser('Data Preparation')
    parser.add_argument('--task', type=str, required=True,
                        choices=['download', 'convert', 'check', 'metadata', 'all'],
                        help='Task to perform')
    # Place to change if dataset name is ModelNet10
    parser.add_argument('--source', type=str, default='data/ModelNet40',
                        help='Source directory for conversion')
    parser.add_argument('--target', type=str, default='data/modelnet40_normal_resampled/',
                        help='Target directory')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample')

    args = parser.parse_args()

    if args.task == 'download' or args.task == 'all':
        download_modelnet()

    if args.task == 'convert':
        if not args.source:
            print("Error: --source required for conversion")
            return
        convert_modelnet(args.source, args.target, args.num_points)

    if args.task == 'check' or args.task == 'all':
        pass
        # check_data_structure(args.target)

    if args.task == 'metadata' or args.task == 'all':
        generate_metadata(args.target)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)

if __name__ == '__main__':
    main()