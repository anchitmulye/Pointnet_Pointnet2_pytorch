"""
Export PointNet++ Semantic Segmentation Models to ONNX format
"""
import argparse
import os
import sys
import torch

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models import pointnet2_sem_seg_onnx


def export_onnx(args):
    """Export PointNet++ semantic segmentation model to ONNX"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = pointnet2_sem_seg_onnx.get_model(
        num_classes=args.num_classes
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input (9 channels: xyz + rgb + normalized xyz)
    dummy_input = torch.randn(1, 9, args.num_point).to(device)

    # Determine output path
    if args.output_path:
        onnx_path = args.output_path
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        experiment_dir = os.path.dirname(checkpoint_dir)
        onnx_dir = os.path.join(experiment_dir, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)

        onnx_filename = f"pointnet2_sem_seg_c{args.num_classes}_n{args.num_point}.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)

    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"Input shape: [batch_size, 9, {args.num_point}]")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
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
    parser = argparse.ArgumentParser('Export PointNet++ Semantic Segmentation to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--num_classes', type=int, default=13,
                        help='Number of semantic classes')
    parser.add_argument('--num_point', type=int, default=4096,
                        help='Number of points in point cloud')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Custom output path for ONNX model')

    args = parser.parse_args()
    export_onnx(args)
