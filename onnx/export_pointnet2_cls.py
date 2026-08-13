"""
Export PointNet++ Classification Models to ONNX format
"""
import argparse
import os
import sys
import torch

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from models import pointnet2_cls_ssg_onnx, pointnet2_cls_msg_onnx
from onnx_utils import default_opset


def export_onnx(args):
    """Export PointNet++ classification model to ONNX"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if args.model == 'pointnet2_cls_ssg':
        model_module = pointnet2_cls_ssg_onnx
    elif args.model == 'pointnet2_cls_msg':
        model_module = pointnet2_cls_msg_onnx
    else:
        raise ValueError(f"Model {args.model} not supported")

    # Create model
    model = model_module.get_model(
        num_class=args.num_category,
        normal_channel=args.use_normals
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input
    in_channels = 6 if args.use_normals else 3
    dummy_input = torch.randn(1, in_channels, args.num_point).to(device)

    # Determine output path
    if args.output_path:
        onnx_path = args.output_path
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        experiment_dir = os.path.dirname(checkpoint_dir)
        onnx_dir = os.path.join(experiment_dir, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)

        model_name = args.model
        onnx_filename = f"{model_name}_c{args.num_category}_n{args.num_point}.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)

    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"Input shape: [batch_size, {in_channels}, {args.num_point}]")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
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
    parser = argparse.ArgumentParser('Export PointNet++ Classification to ONNX')
    parser.add_argument('--model', type=str, required=True,
                        choices=['pointnet2_cls_ssg', 'pointnet2_cls_msg'],
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--num_category', type=int, default=40,
                        help='Number of classes [10, 40]')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='Number of points in point cloud')
    parser.add_argument('--use_normals', action='store_true', default=False,
                        help='Use normals (6 channels instead of 3)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Custom output path for ONNX model')

    args = parser.parse_args()
    export_onnx(args)
