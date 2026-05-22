"""
Generate metadata files for ModelNet dataset
Creates: modelnet40_shape_names.txt, modelnet40_train.txt, modelnet40_test.txt
"""
import os
import glob
import argparse

def generate_metadata(data_root, train_ratio=0.8):
    """Generate metadata files for ModelNet dataset"""

    print("=" * 60)
    print("Generating Metadata Files")
    print("=" * 60)
    print(f"Data root: {data_root}")

    if not os.path.exists(data_root):
        print(f"\nERROR: Directory not found: {data_root}")
        return False

    # Find all class directories (exclude metadata files)
    class_dirs = [d for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d))
                  and not d.startswith('.')
                  and d not in ['__pycache__']]

    class_dirs = sorted(class_dirs)

    if not class_dirs:
        print(f"\nERROR: No class directories found in {data_root}")
        print("Expected structure: data_root/class_name/*.txt")
        return False

    print(f"\nFound {len(class_dirs)} classes")
    print(f"Classes: {class_dirs[:5]} ...")

    # Generate modelnet40_shape_names.txt
    shape_names_file = os.path.join(data_root, 'modelnet40_shape_names.txt')
    with open(shape_names_file, 'w') as f:
        for cls in class_dirs:
            f.write(cls + '\n')

    print(f"\nCreated: modelnet40_shape_names.txt")

    # Generate train and test splits
    train_files = []
    test_files = []

    print("\nProcessing classes:")
    for cls in class_dirs:
        cls_path = os.path.join(data_root, cls)

        # Find all .txt files in this class
        files = glob.glob(os.path.join(cls_path, "*.txt"))
        files = sorted([os.path.basename(f).replace('.txt', '') for f in files])

        if not files:
            print(f"  {cls}: No .txt files found")
            continue

        print(f"  {cls}: {len(files)} files")

        # Split: first train_ratio% train, rest test
        split_idx = int(len(files) * train_ratio)
        if split_idx == 0:
            split_idx = 1  # At least 1 training sample

        train_files.extend(files[:split_idx])
        test_files.extend(files[split_idx:])

    # Write train split
    train_file = os.path.join(data_root, 'modelnet40_train.txt')
    with open(train_file, 'w') as f:
        for fname in train_files:
            f.write(fname + '\n')

    print(f"\nCreated: modelnet40_train.txt ({len(train_files)} samples)")

    # Write test split
    test_file = os.path.join(data_root, 'modelnet40_test.txt')
    with open(test_file, 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')

    print(f"Created: modelnet40_test.txt ({len(test_files)} samples)")

    print("\n" + "=" * 60)
    print("Metadata Generation Complete!")
    print("=" * 60)
    print(f"  Classes: {len(class_dirs)}")
    print(f"  Train samples: {len(train_files)}")
    print(f"  Test samples: {len(test_files)}")
    print(f"  Train/Test ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")
    print("\nYou can now start training:")
    print("  python train_classification.py --model pointnet_cls")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate ModelNet Metadata Files')
    parser.add_argument('--data_root', type=str,
                        default='data/ModelNet40',
                        help='Path to ModelNet data directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    args = parser.parse_args()

    success = generate_metadata(args.data_root, args.train_ratio)

    if not success:
        print("\nFailed to generate metadata files")
        print("\nMake sure your data directory contains class folders with .txt files:")
        print("  data/modelnet40_normal_resampled/")
        print("    ├── airplane/")
        print("    │   ├── airplane_0001.txt")
        print("    │   └── ...")
        print("    ├── bathtub/")
        print("    └── ...")