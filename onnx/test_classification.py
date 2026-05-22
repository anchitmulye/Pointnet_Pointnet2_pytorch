"""
ONNX Runtime Inference Test for PointNet Classification
Tests ONNX model and calculates accuracy on ModelNet test set

Usage:
    python test_onnx_classification.py --onnx_path ../log/classification/test/onnx/pointnet_cls_c10_n1024.onnx --num_category 10
"""

import os
import sys
import argparse
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from data_utils.ModelNetDataLoader import ModelNetDataLoader
import torch.utils.data


def test_onnx_model(onnx_path, num_category=10, num_point=1024, use_normals=False, batch_size=24):
    """
    Test ONNX model and calculate accuracy

    Args:
        onnx_path: Path to ONNX model file
        num_category: Number of classes (10 or 40)
        num_point: Number of points in point cloud
        use_normals: Whether model uses normals (6 channels) or xyz only (3 channels)
        batch_size: Batch size for inference
    """

    print("="*80)
    print("ONNX CLASSIFICATION MODEL - INFERENCE TEST")
    print("="*80)
    print(f"Model: {onnx_path}")
    print(f"Number of classes: {num_category}")
    print(f"Number of points: {num_point}")
    print(f"Use normals: {use_normals}")
    print(f"Batch size: {batch_size}")
    print("="*80)

    # Validate ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        sys.exit(1)

    # Load ONNX model
    print("\nLoading ONNX model...")
    session = ort.InferenceSession(onnx_path)

    # Get model input/output info
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"Input name: {input_name}")
    print(f"Output names: {output_names}")

    # Load test data
    print("\nLoading test data...")
    data_path = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled/')

    test_dataset = ModelNetDataLoader(
        root=data_path,
        args=type('Args', (), {
            'num_category': num_category,
            'num_point': num_point,
            'use_normals': use_normals,
            'use_uniform_sample': False,
            'process_data': False
        })(),
        split='test'
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Test samples: {len(test_dataset)}")

    # Run inference
    print("\nRunning inference...")
    mean_correct = []
    class_acc = np.zeros((num_category, 3))

    for points, target in tqdm(test_loader, desc="Testing"):
        # Convert to numpy
        points = points.numpy()  # (batch_size, num_points, channels)
        target = target.numpy()

        # Transpose to (batch_size, channels, num_points) for model input
        points = np.transpose(points, (0, 2, 1)).astype(np.float32)

        # Run ONNX inference
        outputs = session.run(None, {input_name: points})
        logits = outputs[0]  # (batch_size, num_classes)

        # Get predictions
        pred = np.argmax(logits, axis=1)

        # Calculate accuracy
        correct = (pred == target)
        mean_correct.append(np.mean(correct))

        # Per-class accuracy
        for cat in np.unique(target):
            classacc = correct[target == cat]
            class_acc[cat, 0] += np.sum(classacc)
            class_acc[cat, 1] += len(classacc)

    # Calculate final metrics
    instance_acc = np.mean(mean_correct)
    class_acc[:, 2] = class_acc[:, 0] / (class_acc[:, 1] + 1e-8)
    class_avg_acc = np.mean(class_acc[:, 2])

    # Print results
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    print(f"Instance Accuracy: {instance_acc:.4f} ({instance_acc*100:.2f}%)")
    print(f"Class Average Accuracy: {class_avg_acc:.4f} ({class_avg_acc*100:.2f}%)")
    print("="*80)

    return instance_acc, class_avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ONNX Classification Inference Test')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--num_category', type=int, default=10,
                        help='Number of classes (10 or 40)')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='Number of points')
    parser.add_argument('--use_normals', action='store_true', default=False,
                        help='Use normals (6 channels)')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for inference')

    args = parser.parse_args()

    # Run test
    test_onnx_model(
        onnx_path=args.onnx_path,
        num_category=args.num_category,
        num_point=args.num_point,
        use_normals=args.use_normals,
        batch_size=args.batch_size
    )
