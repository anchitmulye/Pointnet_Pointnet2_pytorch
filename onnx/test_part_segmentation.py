"""
ONNX Runtime Inference Test for PointNet Part Segmentation
Tests ONNX model and calculates IoU on ShapeNet test set

Usage:
    python test_onnx_partseg.py --onnx_path ../log/part_seg/pointnet_part_seg/onnx/pointnet_part_seg_n2048.onnx
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

from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch.utils.data

# ShapeNet part segmentation classes
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = np.eye(num_classes)[y]
    return new_y


def test_onnx_model(onnx_path, num_classes=50, num_category=16, num_point=2048, normal=False, batch_size=24, num_votes=3):
    """
    Test ONNX model and calculate part segmentation IoU

    Args:
        onnx_path: Path to ONNX model file
        num_classes: Number of part classes (default: 50 for ShapeNet)
        num_category: Number of object categories (default: 16 for ShapeNet)
        num_point: Number of points in point cloud
        normal: Whether model uses normals
        batch_size: Batch size for inference
        num_votes: Number of votes for aggregation
    """

    print("="*80)
    print("ONNX PART SEGMENTATION MODEL - INFERENCE TEST")
    print("="*80)
    print(f"Model: {onnx_path}")
    print(f"Number of part classes: {num_classes}")
    print(f"Number of object categories: {num_category}")
    print(f"Number of points: {num_point}")
    print(f"Use normals: {normal}")
    print(f"Batch size: {batch_size}")
    print(f"Number of votes: {num_votes}")
    print("="*80)

    # Validate ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        sys.exit(1)

    # Load ONNX model
    print("\nLoading ONNX model...")
    session = ort.InferenceSession(onnx_path)

    # Get model input/output info
    inputs = session.get_inputs()
    print(f"Model inputs: {[inp.name for inp in inputs]}")
    print(f"Model outputs: {[out.name for out in session.get_outputs()]}")

    # Load test data
    print("\nLoading test data...")
    data_path = os.path.join(ROOT_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    test_dataset = PartNormalDataset(
        root=data_path,
        npoints=num_point,
        split='test',
        normal_channel=normal
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
    shape_ious = {cat: [] for cat in seg_classes.keys()}

    for batch_id, (points, label, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
        batch_size_actual = points.shape[0]
        num_point_actual = points.shape[1]

        # Voting
        vote_pool = np.zeros((batch_size_actual, num_point_actual, num_classes))

        for v in range(num_votes):
            # Convert to numpy
            points_np = points.numpy()  # (batch, num_points, channels)
            label_np = label.numpy()  # (batch,)

            # Transpose points to (batch, channels, num_points)
            points_input = np.transpose(points_np, (0, 2, 1)).astype(np.float32)

            # Create one-hot category label (batch, num_category, num_points)
            label_one_hot = to_categorical(label_np, num_category)  # (batch, num_category)
            label_one_hot = np.repeat(label_one_hot[:, :, np.newaxis], num_point_actual, axis=2).astype(np.float32)

            # Prepare inputs
            input_dict = {
                inputs[0].name: points_input,
                inputs[1].name: label_one_hot
            }

            # Run ONNX inference
            outputs = session.run(None, input_dict)
            seg_pred = outputs[0]  # (batch, num_classes, num_points) or (batch, num_points, num_classes)

            # Handle different output formats
            if len(seg_pred.shape) == 3 and seg_pred.shape[1] == num_classes:
                # (batch, num_classes, num_points) -> (batch, num_points, num_classes)
                seg_pred = np.transpose(seg_pred, (0, 2, 1))

            vote_pool += seg_pred

        # Get final predictions
        seg_pred = np.argmax(vote_pool, axis=2)  # (batch, num_points)

        # Calculate IoU
        target = target.numpy()  # (batch, num_points)
        label_np = label.numpy()

        for shape_idx in range(batch_size_actual):
            cur_pred = seg_pred[shape_idx]  # (num_points,)
            cur_gt = target[shape_idx]  # (num_points,)
            cur_label = label_np[shape_idx]

            # Get category
            cat = seg_label_to_cat[cur_gt[0]]
            part_ious = []

            # Calculate IoU for each part in this category
            for part in seg_classes[cat]:
                pred_part = (cur_pred == part)
                gt_part = (cur_gt == part)
                union = np.sum(pred_part | gt_part)
                intersection = np.sum(pred_part & gt_part)

                if union == 0:
                    iou = 1.0  # No part, perfect prediction
                else:
                    iou = intersection / float(union)
                part_ious.append(iou)

            shape_ious[cat].append(np.mean(part_ious))

    # Calculate metrics
    all_shape_ious = []
    for cat in shape_ious.keys():
        if len(shape_ious[cat]) > 0:
            all_shape_ious += shape_ious[cat]
            shape_ious[cat] = np.mean(shape_ious[cat])
        else:
            shape_ious[cat] = 0.0

    instance_avg_iou = np.mean(all_shape_ious)
    class_avg_iou = np.mean(list(shape_ious.values()))

    # Print results
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    print(f"Instance Average IoU: {instance_avg_iou:.4f} ({instance_avg_iou*100:.2f}%)")
    print(f"Class Average IoU: {class_avg_iou:.4f} ({class_avg_iou*100:.2f}%)")
    print("\nPer-category IoU:")
    for cat in sorted(shape_ious.keys()):
        print(f"  {cat:12s}: {shape_ious[cat]:.4f}")
    print("="*80)

    return instance_avg_iou, class_avg_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ONNX Part Segmentation Inference Test')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--num_classes', type=int, default=50,
                        help='Number of part classes')
    parser.add_argument('--num_category', type=int, default=16,
                        help='Number of object categories')
    parser.add_argument('--num_point', type=int, default=2048,
                        help='Number of points')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Use normals')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for inference')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='Number of votes for aggregation')

    args = parser.parse_args()

    # Run test
    test_onnx_model(
        onnx_path=args.onnx_path,
        num_classes=args.num_classes,
        num_category=args.num_category,
        num_point=args.num_point,
        normal=args.normal,
        batch_size=args.batch_size,
        num_votes=args.num_votes
    )
