"""
Main Entry Point for PointNet/PointNet++ Training and Inference
Single unified script to run all tasks
"""

import sys
import argparse
import torch

def train_classification(args):
    """Train classification model"""
    from train_classification import main as train_main, parse_args
    sys.argv = ['train_classification.py',
                '--model', args.model,
                '--epoch', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--num_category', str(args.num_category),
                '--num_point', str(args.num_points),
                '--gpu', args.gpu]
    if args.use_cpu:
        sys.argv.append('--use_cpu')
    train_args = parse_args()
    train_main(train_args)

def test_classification(args):
    """Test classification model"""
    from test_classification import main as test_main
    sys.argv = ['test_classification.py',
                '--log_dir', args.checkpoint]
    test_main()

def train_partseg(args):
    """Train part segmentation model"""
    from train_partseg import main as train_main, parse_args
    sys.argv = ['train_partseg.py',
                '--epoch', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--gpu', args.gpu]
    if args.use_cpu:
        sys.argv.append('--use_cpu')
    train_args = parse_args()
    train_main(train_args)

def test_partseg(args):
    """Test part segmentation model"""
    from test_partseg import main as test_main
    sys.argv = ['test_partseg.py',
                '--log_dir', args.checkpoint]
    test_main()

def train_semseg(args):
    """Train semantic segmentation model"""
    from train_semseg import main as train_main, parse_args
    sys.argv = ['train_semseg.py',
                '--epoch', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--gpu', args.gpu]
    if args.use_cpu:
        sys.argv.append('--use_cpu')
    train_args = parse_args()
    train_main(train_args)

def test_semseg(args):
    """Test semantic segmentation model"""
    from test_semseg import main as test_main
    sys.argv = ['test_semseg.py',
                '--log_dir', args.checkpoint]
    test_main()

def export_onnx(args):
    """Export model to ONNX"""
    # Auto-detect num_category from checkpoint if not specified for classification
    if args.task_type == 'classification' and args.num_category is None:
        print(f"Auto-detecting number of categories from checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            # Get shape from fc3.weight (last layer)
            fc3_weight = checkpoint['model_state_dict'].get('fc3.weight')
            if fc3_weight is not None:
                args.num_category = fc3_weight.shape[0]
                print(f"Detected {args.num_category} categories from checkpoint")

    if args.task_type == 'classification':
        from export_onnx_classification import main as export_main
        sys.argv = ['export_onnx_classification.py',
                    '--model', args.model,
                    '--checkpoint', args.checkpoint,
                    '--num_category', str(args.num_category)]
    elif args.task_type == 'partseg':
        from export_onnx_partseg import main as export_main
        sys.argv = ['export_onnx_partseg.py',
                    '--checkpoint', args.checkpoint]
    elif args.task_type == 'semseg':
        from export_onnx_semseg import main as export_main
        sys.argv = ['export_onnx_semseg.py',
                    '--checkpoint', args.checkpoint]
    export_main()

def main():
    parser = argparse.ArgumentParser('PointNet/PointNet++ Main Runner')

    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['train', 'test', 'export'],
                        help='Task to perform: train, test, or export')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'partseg', 'semseg'],
                        help='Type of task')

    # Training arguments
    parser.add_argument('--model', type=str, default='pointnet_cls',
                        help='Model name')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size')
    parser.add_argument('--num_category', type=int, default=None,
                        choices=[10, 40],
                        help='Number of categories (10 or 40, auto-detect from checkpoint if not specified)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use CPU mode')

    # Testing/Export arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint or log directory')

    args = parser.parse_args()

    # Route to appropriate task
    if args.task == 'train':
        if args.task_type == 'classification':
            train_classification(args)
        elif args.task_type == 'partseg':
            train_partseg(args)
        elif args.task_type == 'semseg':
            train_semseg(args)

    elif args.task == 'test':
        if not args.checkpoint:
            print("Error: --checkpoint required for testing")
            return
        if args.task_type == 'classification':
            test_classification(args)
        elif args.task_type == 'partseg':
            test_partseg(args)
        elif args.task_type == 'semseg':
            test_semseg(args)

    elif args.task == 'export':
        if not args.checkpoint:
            print("Error: --checkpoint required for export")
            return
        export_onnx(args)

if __name__ == '__main__':
    main()