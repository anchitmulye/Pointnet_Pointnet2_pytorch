"""
Export PointNet++ Part Segmentation Models to ONNX format
"""
import argparse
import os
import sys
import torch

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models import pointnet2_part_seg_ssg_onnx, pointnet2_part_seg_msg_onnx


def export_onnx(args):
    """Export PointNet++ part segmentation model to ONNX"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select model type
    if args.model == 'pointnet2_part_seg_msg':
        model_module = pointnet2_part_seg_msg_onnx
    else:
        model_module = pointnet2_part_seg_ssg_onnx

    # Create model
    model = model_module.get_model(
        num_classes=args.num_parts,
        normal_channel=args.use_normals
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy inputs
    in_channels = 6 if args.use_normals else 3
    dummy_points = torch.randn(1, in_channels, args.num_point).to(device)
    dummy_label = torch.zeros(1, 16).to(device)  # One-hot class label
    dummy_label[0, 0] = 1  # First class

    # Determine output path
    if args.output_path:
        onnx_path = args.output_path
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        experiment_dir = os.path.dirname(checkpoint_dir)
        onnx_dir = os.path.join(experiment_dir, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)

        onnx_filename = f"pointnet2_part_seg_n{args.num_point}.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)

    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"Input shape: points=[batch_size, {in_channels}, {args.num_point}], label=[batch_size, 16]")

    torch.onnx.export(
        model,
        (dummy_points, dummy_label),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['points', 'class_label'],
        output_names=['output', 'features'],
        dynamic_axes={
            'points': {0: 'batch_size'},
            'class_label': {0: 'batch_size'},
            'output': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )

    # Verify the model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"ONNX model exported successfully!")
    print(f"Model saved to: {onnx_path}")
    print(f"Model verification passed")

    return onnx_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export PointNet++ Part Segmentation to ONNX')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_ssg',
                        choices=['pointnet2_part_seg_ssg', 'pointnet2_part_seg_msg'],
                        help='Model type (SSG or MSG)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--num_parts', type=int, default=50,
                        help='Number of part classes')
    parser.add_argument('--num_point', type=int, default=2048,
                        help='Number of points in point cloud')
    parser.add_argument('--use_normals', action='store_true', default=False,
                        help='Use normals (6 channels instead of 3)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Custom output path for ONNX model')

    args = parser.parse_args()
    export_onnx(args)
