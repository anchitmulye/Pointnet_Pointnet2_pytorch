"""
ONNX Runtime Inference Test for PointNet Semantic Segmentation
Tests ONNX model and calculates accuracy/IoU on S3DIS test set

Usage:
    python test_onnx_semseg.py --onnx_path ../log/sem_seg/pointnet_sem_seg/onnx/pointnet_sem_seg_c13_n4096.onnx --test_area 5
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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
import provider

# S3DIS classes
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def test_onnx_model(onnx_path, num_classes=13, num_point=4096, batch_size=32, test_area=5, num_votes=3):
    """
    Test ONNX model and calculate semantic segmentation accuracy

    Args:
        onnx_path: Path to ONNX model file
        num_classes: Number of semantic classes (default: 13 for S3DIS)
        num_point: Number of points per block
        batch_size: Batch size for inference
        test_area: Area for testing (1-6)
        num_votes: Number of votes for aggregation
    """

    print("="*80)
    print("ONNX SEMANTIC SEGMENTATION MODEL - INFERENCE TEST")
    print("="*80)
    print(f"Model: {onnx_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of points: {num_point}")
    print(f"Batch size: {batch_size}")
    print(f"Test area: {test_area}")
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
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"Input name: {input_name}")
    print(f"Output names: {output_names}")

    # Load test data
    print("\nLoading test data...")
    data_path = os.path.join(ROOT_DIR, 'data/stanford_indoor3d/')

    test_dataset = ScannetDatasetWholeScene(
        root=data_path,
        block_points=num_point,
        split='test',
        test_area=test_area,
        stride=0.5,
        block_size=1.0,
        padding=0.001
    )

    print(f"Test scenes: {len(test_dataset)}")

    # Run inference
    print("\nRunning inference...")
    scene_id = test_dataset.file_list
    num_batches = test_dataset.batch_amount

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]
    total_iou_deno_class = [0 for _ in range(num_classes)]

    for batch_idx in tqdm(range(num_batches), desc="Testing"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_dataset))

        # Process batch
        whole_scene_data = []
        whole_scene_label = []

        for idx in range(start_idx, end_idx):
            data, label = test_dataset[idx]
            whole_scene_data.append(data)
            whole_scene_label.append(label)

        whole_scene_data = np.concatenate(whole_scene_data, axis=0)  # (N, num_point, 9)
        whole_scene_label = np.concatenate(whole_scene_label, axis=0)  # (N, num_point)

        # Initialize voting pool
        vote_label_pool = np.zeros((whole_scene_label.shape[0], whole_scene_label.shape[1], num_classes))

        # Voting
        for vote_idx in range(num_votes):
            # Shuffle point order for voting
            scene_data = np.copy(whole_scene_data)
            for i in range(scene_data.shape[0]):
                scene_data[i, :, 0:3] = provider.rotate_point_cloud_z(scene_data[i, :, 0:3])

            # Transpose to (batch, 9, num_point) for model input
            scene_data_input = np.transpose(scene_data, (0, 2, 1)).astype(np.float32)

            # Run ONNX inference
            outputs = session.run(None, {input_name: scene_data_input})
            seg_pred = outputs[0]  # (batch, num_point, num_classes)

            # Accumulate votes
            vote_label_pool += seg_pred

        # Get final predictions
        pred_label = np.argmax(vote_label_pool, axis=2)  # (batch, num_point)

        # Calculate accuracy
        for b in range(pred_label.shape[0]):
            for n in range(pred_label.shape[1]):
                gt_label = int(whole_scene_label[b, n])
                pred = int(pred_label[b, n])

                total_seen += 1
                if pred == gt_label:
                    total_correct += 1

                total_seen_class[gt_label] += 1
                total_correct_class[gt_label] += (pred == gt_label)
                total_iou_deno_class[gt_label] += 1

                # Count false positives
                if pred != gt_label:
                    total_iou_deno_class[pred] += 1

    # Calculate metrics
    overall_acc = total_correct / float(total_seen)

    class_accuracies = []
    class_ious = []
    for i in range(num_classes):
        if total_seen_class[i] == 0:
            acc = 0
        else:
            acc = total_correct_class[i] / float(total_seen_class[i])
        class_accuracies.append(acc)

        if total_iou_deno_class[i] == 0:
            iou = 0
        else:
            iou = total_correct_class[i] / float(total_iou_deno_class[i])
        class_ious.append(iou)

    mean_class_acc = np.mean(class_accuracies)
    mean_class_iou = np.mean(class_ious)

    # Print results
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Mean Class Accuracy: {mean_class_acc:.4f} ({mean_class_acc*100:.2f}%)")
    print(f"Mean Class IoU: {mean_class_iou:.4f} ({mean_class_iou*100:.2f}%)")
    print("\nPer-class IoU:")
    for i, cls in enumerate(classes):
        print(f"  {cls:12s}: {class_ious[i]:.4f}")
    print("="*80)

    return overall_acc, mean_class_acc, mean_class_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ONNX Semantic Segmentation Inference Test')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--num_classes', type=int, default=13,
                        help='Number of semantic classes')
    parser.add_argument('--num_point', type=int, default=4096,
                        help='Number of points per block')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--test_area', type=int, default=5,
                        help='Area for testing (1-6)')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='Number of votes for aggregation')

    args = parser.parse_args()

    # Run test
    test_onnx_model(
        onnx_path=args.onnx_path,
        num_classes=args.num_classes,
        num_point=args.num_point,
        batch_size=args.batch_size,
        test_area=args.test_area,
        num_votes=args.num_votes
    )
