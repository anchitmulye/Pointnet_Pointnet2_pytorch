"""
Convert Original ModelNet10 (.off files) to PointNet Format
Converts your data structure to the format expected by the training code
"""
import os
import numpy as np
from tqdm import tqdm
import argparse


def read_off(file_path):
    """Read .off file and return vertices and faces"""
    with open(file_path, 'r') as f:
        # Read header
        first_line = f.readline().strip()
        if first_line == 'OFF':
            # Standard OFF format
            second_line = f.readline().strip()
        else:
            # Sometimes OFF and counts are on same line
            second_line = first_line[3:].strip()

        # Parse counts
        parts = second_line.split()
        n_verts = int(parts[0])
        n_faces = int(parts[1]) if len(parts) > 1 else 0

        # Read vertices
        verts = []
        for i in range(n_verts):
            line = f.readline().strip().split()
            verts.append([float(x) for x in line[:3]])

        verts = np.array(verts, dtype=np.float32)

    return verts


def sample_points_from_mesh(vertices, num_points=1024):
    """
    Sample points from mesh vertices
    Uses random sampling with replacement
    """
    if len(vertices) >= num_points:
        # Random sampling without replacement
        indices = np.random.choice(len(vertices), num_points, replace=False)
    else:
        # Random sampling with replacement if not enough vertices
        indices = np.random.choice(len(vertices), num_points, replace=True)

    points = vertices[indices]
    return points


def compute_normals(points):
    """
    Compute simple normals (towards centroid)
    For proper normals, we'd need face information
    """
    centroid = np.mean(points, axis=0)
    normals = points - centroid
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normals = normals / norms
    return normals


def convert_modelnet(input_root, output_root, num_points=1024, num_category=40):
    """
    Convert ModelNet .off files to .txt format

    Args:
        input_root: Path to original ModelNet (contains class/train and class/test)
        output_root: Path to output directory
        num_points: Number of points to sample per model
    """

    print("=" * 70)
    print("Converting ModelNet from .off to .txt format")
    print("=" * 70)
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Points per model: {num_points}")
    print()

    # Create output directory
    os.makedirs(output_root, exist_ok=True)

    # Get all class names
    classes = [d for d in os.listdir(input_root)
               if os.path.isdir(os.path.join(input_root, d))
               and not d.startswith('.')]
    classes = sorted(classes)

    print(f"Found {len(classes)} classes: {classes}\n")

    train_files = []
    test_files = []

    # Process each class
    for cls in classes:
        print(f"Processing class: {cls}")

        # Create output directory for this class
        cls_output_dir = os.path.join(output_root, cls)
        os.makedirs(cls_output_dir, exist_ok=True)

        # Process training data
        train_dir = os.path.join(input_root, cls, 'train')
        if os.path.exists(train_dir):
            off_files = [f for f in os.listdir(train_dir) if f.endswith('.off')]
            print(f"  Train: {len(off_files)} files")

            for off_file in tqdm(off_files, desc=f"  Converting train", leave=False):
                try:
                    # Read OFF file
                    off_path = os.path.join(train_dir, off_file)
                    vertices = read_off(off_path)

                    # Sample points
                    points = sample_points_from_mesh(vertices, num_points)

                    # Compute normals
                    normals = compute_normals(points)

                    # Combine points and normals (x, y, z, nx, ny, nz)
                    points_with_normals = np.hstack([points, normals])

                    # Save as .txt
                    txt_filename = off_file.replace('.off', '.txt')
                    txt_path = os.path.join(cls_output_dir, txt_filename)
                    np.savetxt(txt_path, points_with_normals, delimiter=',', fmt='%.6f')

                    # Record filename (without extension)
                    train_files.append(txt_filename.replace('.txt', ''))

                except Exception as e:
                    print(f"    Error processing {off_file}: {e}")

        # Process test data
        test_dir = os.path.join(input_root, cls, 'test')
        if os.path.exists(test_dir):
            off_files = [f for f in os.listdir(test_dir) if f.endswith('.off')]
            print(f"  Test:  {len(off_files)} files")

            for off_file in tqdm(off_files, desc=f"  Converting test", leave=False):
                try:
                    # Read OFF file
                    off_path = os.path.join(test_dir, off_file)
                    vertices = read_off(off_path)

                    # Sample points
                    points = sample_points_from_mesh(vertices, num_points)

                    # Compute normals
                    normals = compute_normals(points)

                    # Combine points and normals
                    points_with_normals = np.hstack([points, normals])

                    # Save as .txt
                    txt_filename = off_file.replace('.off', '.txt')
                    txt_path = os.path.join(cls_output_dir, txt_filename)
                    np.savetxt(txt_path, points_with_normals, delimiter=',', fmt='%.6f')

                    # Record filename (without extension)
                    test_files.append(txt_filename.replace('.txt', ''))

                except Exception as e:
                    print(f"    Error processing {off_file}: {e}")

        print()

    # Generate metadata files
    print("Generating metadata files...")

    # modelnet10_shape_names.txt
    if num_category == 10:
        file_prefix = "modelnet10"
    else:
        file_prefix = "modelnet40"
    shape_names_file = os.path.join(output_root, f"{file_prefix}_shape_names.txt")
    with open(shape_names_file, 'w') as f:
        for cls in classes:
            f.write(cls + '\n')
    print(f"  Created: {file_prefix}_shape_names.txt")

    train_file = os.path.join(output_root, f"{file_prefix}_train.txt")
    with open(train_file, 'w') as f:
        for filename in train_files:
            f.write(filename + '\n')
    print(f"  Created: {file_prefix}_train.txt ({len(train_files)} samples)")

    test_file = os.path.join(output_root, f"{file_prefix}_test.txt")
    with open(test_file, 'w') as f:
        for filename in test_files:
            f.write(filename + '\n')
    print(f"  Created: {file_prefix}_test.txt ({len(test_files)} samples)")

    print()
    print("=" * 70)
    print("  Conversion Complete!")
    print("=" * 70)
    print(f"  Classes: {len(classes)}")
    print(f"  Train samples: {len(train_files)}")
    print(f"  Test samples: {len(test_files)}")
    print(f"  Output directory: {output_root}")
    print()
    print("You can now train with:")
    print(f"  python train_classification.py \\")
    print(f"    --model pointnet_cls \\")
    print(f"    --num_category {num_category}\\")
    print(f"    --use_normals \\")
    print(f"    --log_dir pointnet_modelnet10")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert ModelNet40 to Training Format')
    parser.add_argument('--input_root', type=str, default='data/ModelNet40',
                        help='Path to original ModelNet40 directory')
    parser.add_argument('--output_root', type=str, default='data/modelnet40_normal_resampled',
                        help='Path to output directory')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample per model')
    args = parser.parse_args()

    if not os.path.exists(args.input_root):
        print(f" ERROR: Input directory not found: {args.input_root}")
        print("\nExpected structure:")
        print("  data/ModelNet40/")
        print("    ├── bathtub/")
        print("    │   ├── train/")
        print("    │   │   ├── bathtub_0001.off")
        print("    │   │   └── ...")
        print("    │   └── test/")
        print("    │       ├── bathtub_0107.off")
        print("    │       └── ...")
        print("    ├── bed/")
        print("    └── ...")
        exit(1)

    convert_modelnet(args.input_root, args.output_root, args.num_points)