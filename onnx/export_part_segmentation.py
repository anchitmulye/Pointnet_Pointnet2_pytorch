"""
ONNX Export Script for PointNet/PointNet++ Part Segmentation Models
Export trained PyTorch models to ONNX format without modifying original model code
"""

import os
import sys
import torch
import argparse
import importlib
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('ONNX Export for Part Segmentation')
    parser.add_argument('--model', type=str, required=True,
                        help='model name (e.g., pointnet_part_seg, pointnet2_part_seg_ssg, pointnet2_part_seg_msg)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint file (.pth)')
    parser.add_argument('--num_classes', type=int, default=50,
                        help='number of part classes (default: 50 for ShapeNet)')
    parser.add_argument('--num_category', type=int, default=16,
                        help='number of object categories (default: 16 for ShapeNet)')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='use normals')
    parser.add_argument('--num_point', type=int, default=2048,
                        help='number of points in point cloud')
    parser.add_argument('--output_path', type=str, default=None,
                        help='output ONNX file path (default: model_name.onnx)')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true', default=False,
                        help='simplify ONNX model using onnx-simplifier')
    return parser.parse_args()


def export_to_onnx(args):
    """Export part segmentation model to ONNX format"""

    # Determine input channels
    in_channels = 6 if args.normal else 3

    # Load the model architecture
    print(f"Loading model architecture: {args.model}")
    model_module = importlib.import_module(args.model)
    classifier = model_module.get_model(args.num_classes, normal_channel=args.normal)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'instance_avg_iou' in checkpoint:
            print(f"Instance avg IoU: {checkpoint['instance_avg_iou']}")
        if 'class_avg_iou' in checkpoint:
            print(f"Class avg IoU: {checkpoint['class_avg_iou']}")
    else:
        classifier.load_state_dict(checkpoint)

    # Set model to evaluation mode
    classifier.eval()

    # Create dummy inputs
    # Point cloud: (batch_size, channels, num_points)
    dummy_points = torch.randn(1, in_channels, args.num_point)
    # Class label: (batch_size, 1, num_points) - one-hot encoded category
    dummy_label = torch.zeros(1, args.num_category, args.num_point)
    dummy_label[0, 0, :] = 1  # Set first category as example

    # Set output path
    if args.output_path is None:
        # Create onnx directory in the same location as checkpoint
        checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))  # Go up from checkpoints/ to experiment dir
        onnx_dir = os.path.join(checkpoint_dir, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)

        output_filename = f"{args.model}_n{args.num_point}"
        if args.normal:
            output_filename += "_normal"
        output_filename += ".onnx"
        output_path = os.path.join(onnx_dir, output_filename)
    else:
        output_path = args.output_path

    print(f"\nExporting to ONNX...")
    print(f"Point cloud shape: {dummy_points.shape}")
    print(f"Category label shape: {dummy_label.shape}")
    print(f"Output path: {output_path}")

    # Export to ONNX
    torch.onnx.export(
        classifier,
        (dummy_points, dummy_label),
        output_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=['point_cloud', 'category_label'],
        output_names=['seg_pred', 'features'],
        dynamic_axes={
            'point_cloud': {0: 'batch_size'},
            'category_label': {0: 'batch_size'},
            'seg_pred': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported successfully to: {output_path}")

    # Optional: Simplify the ONNX model
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            print("\nSimplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            model_simplified, check = simplify(onnx_model)

            if check:
                simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                onnx.save(model_simplified, simplified_path)
                print(f"implified ONNX model saved to: {simplified_path}")
            else:
                print("Simplification failed - validation error")

        except ImportError:
            print("onnx-simplifier not installed. Install with: pip install onnx-simplifier")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model verification passed")

        # Print model info
        print("\n--- Model Information ---")
        print(f"IR Version: {onnx_model.ir_version}")
        print(f"Producer: {onnx_model.producer_name}")
        print(f"Opset Version: {onnx_model.opset_import[0].version}")
        print(f"Inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")

    except ImportError:
        print("onnx not installed. Install with: pip install onnx")
    except Exception as e:
        print(f"ONNX verification failed: {str(e)}")

    return output_path


def main():
    args = parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Export model
    output_path = export_to_onnx(args)

    print("\n" + "="*60)
    print("Export completed successfully!")
    print("="*60)
    print(f"\nTo use the ONNX model:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{output_path}')")
    print(f"  outputs = session.run(None, {{")
    print(f"      'point_cloud': point_data,")
    print(f"      'category_label': category_data")
    print(f"  }})")
    print()


if __name__ == '__main__':
    main()
