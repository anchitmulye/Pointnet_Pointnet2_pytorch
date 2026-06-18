"""
Unified test runner for PointNet / PointNet++ models.
Runs both PyTorch (.pth) and ONNX (.onnx) inference from a single log directory
and prints accuracy side-by-side.

Usage examples:
    # Classification
    python test/test.py --task classification --log_dir pointnet_10 --num_category 10
    python test/test.py --task classification --log_dir pointnet2_ssg_wo_normals --num_category 40
    python test/test.py --task classification --log_dir pointnet2_msg_normals --num_category 40 --use_normals

    # Part segmentation
    python test/test.py --task part_seg --log_dir pointnet2_part_seg_msg

    # Semantic segmentation
    python test/test.py --task sem_seg --log_dir pointnet_sem_seg --num_classes 13
    python test/test.py --task sem_seg --log_dir pointnet2_sem_seg --num_classes 13

    # Skip one backend
    python test/test.py --task classification --log_dir pointnet_10 --num_category 10 --skip_pth
    python test/test.py --task classification --log_dir pointnet_10 --num_category 10 --skip_onnx
"""

import argparse
import importlib
import os
import sys
import glob

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'models'))

# ── ShapeNet part-seg constants ───────────────────────────────────────────────
SEG_CLASSES = {
    'Earphone': [16,17,18], 'Motorbike': [30,31,32,33,34,35], 'Rocket': [41,42,43],
    'Car': [8,9,10,11], 'Laptop': [28,29], 'Cap': [6,7], 'Skateboard': [44,45,46],
    'Mug': [36,37], 'Guitar': [19,20,21], 'Bag': [4,5], 'Lamp': [24,25,26,27],
    'Table': [47,48,49], 'Airplane': [0,1,2,3], 'Pistol': [38,39,40],
    'Chair': [12,13,14,15], 'Knife': [22,23],
}
SEG_LABEL_TO_CAT = {label: cat for cat, labels in SEG_CLASSES.items() for label in labels}

# ── S3DIS sem-seg constants ───────────────────────────────────────────────────
SEM_CLASSES = ['ceiling','floor','wall','beam','column','window','door',
               'table','chair','sofa','bookcase','board','clutter']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_onnx(exp_dir):
    matches = glob.glob(os.path.join(exp_dir, 'onnx', '*.onnx'))
    if not matches:
        return None
    return matches[0]


def find_model_name(exp_dir):
    """Infer model module name from the .py file copied into the log dir."""
    candidates = [f for f in os.listdir(exp_dir)
                  if f.endswith('.py') and f not in ('train_classification.py',
                                                      'pointnet2_utils.py',
                                                      'pointnet_utils.py')]
    if not candidates:
        return None
    return candidates[0].replace('.py', '')


def load_onnx_session(onnx_path):
    import onnxruntime as ort
    return ort.InferenceSession(onnx_path)


def header(text):
    bar = '─' * 60
    print(f'\n{bar}\n  {text}\n{bar}')


def result_line(label, instance_acc, extra=None):
    line = f'  {label:<22}  Instance Acc: {instance_acc*100:6.2f}%'
    if extra:
        line += f'  |  {extra}'
    print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def _cls_loader(args, exp_dir):
    from data_utils.ModelNetDataLoader import ModelNetDataLoader
    data_path = os.path.join(ROOT_DIR, 'data', 'modelnet40_normal_resampled')
    cfg = type('A', (), {
        'num_category': args.num_category,
        'num_point': args.num_point,
        'use_normals': args.use_normals,
        'use_uniform_sample': False,
        'process_data': False,
    })()
    ds = ModelNetDataLoader(root=data_path, args=cfg, split='test')
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4)


def test_cls_pth(args, exp_dir):
    model_name = find_model_name(exp_dir)
    if model_name is None:
        print('  [pth] Could not detect model file in log dir — skipping.')
        return None

    sys.path.insert(0, exp_dir)
    model_mod = importlib.import_module(model_name)
    sys.path.pop(0)

    classifier = model_mod.get_model(args.num_category, normal_channel=args.use_normals)
    classifier.eval()

    ckpt = torch.load(os.path.join(exp_dir, 'checkpoints', 'best_model.pth'),
                      map_location='cpu', weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'])

    loader = _cls_loader(args, exp_dir)
    correct = total = 0
    with torch.no_grad():
        for points, target in tqdm(loader, desc='  [pth] cls', leave=False):
            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            pred_choice = pred.argmax(dim=1)
            correct += pred_choice.eq(target.long()).sum().item()
            total += target.size(0)

    return correct / total


def test_cls_onnx(args, exp_dir):
    onnx_path = find_onnx(exp_dir)
    if onnx_path is None:
        print('  [onnx] No .onnx file found in log/onnx/ — skipping.')
        return None

    session = load_onnx_session(onnx_path)
    input_name = session.get_inputs()[0].name
    loader = _cls_loader(args, exp_dir)

    correct = total = 0
    for points, target in tqdm(loader, desc='  [onnx] cls', leave=False):
        pts = points.numpy().transpose(0, 2, 1).astype(np.float32)
        logits = session.run(None, {input_name: pts})[0]
        pred = np.argmax(logits, axis=1)
        correct += (pred == target.numpy()).sum()
        total += target.size(0)

    return correct / total


def run_classification(args, exp_dir):
    header(f'Classification  |  {os.path.basename(exp_dir)}  |  C{args.num_category}  N{args.num_point}')

    pth_acc = onnx_acc = None

    if not args.skip_pth:
        pth_acc = test_cls_pth(args, exp_dir)
        if pth_acc is not None:
            result_line('[pth]', pth_acc)

    if not args.skip_onnx:
        onnx_acc = test_cls_onnx(args, exp_dir)
        if onnx_acc is not None:
            result_line('[onnx]', onnx_acc)

    if pth_acc is not None and onnx_acc is not None:
        diff = abs(pth_acc - onnx_acc) * 100
        print(f'\n  pth vs onnx: {diff:.2f}pp')


# ─────────────────────────────────────────────────────────────────────────────
# Part segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _partseg_loader(args):
    from data_utils.ShapeNetDataLoader import PartNormalDataset
    root = os.path.join(ROOT_DIR, 'data',
                        'shapenetcore_partanno_segmentation_benchmark_v0_normal')
    ds = PartNormalDataset(root=root, npoints=args.num_point, split='test',
                           normal_channel=args.use_normals)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4)


def _partseg_metrics(pred_labels, targets):
    """Returns (point_acc, mean_iou_per_shape)."""
    all_ious = []
    total_correct = total_seen = 0
    for pred, tgt in zip(pred_labels, targets):
        correct = (pred == tgt).sum()
        total_correct += correct
        total_seen += len(tgt)
        cat = SEG_LABEL_TO_CAT[tgt[0]]
        part_ious = []
        for l in SEG_CLASSES[cat]:
            union = ((pred == l) | (tgt == l)).sum()
            inter = ((pred == l) & (tgt == l)).sum()
            part_ious.append(1.0 if union == 0 else inter / union)
        all_ious.append(np.mean(part_ious))
    return total_correct / total_seen, np.mean(all_ious)


def test_partseg_pth(args, exp_dir):
    model_name = find_model_name(exp_dir)
    if model_name is None:
        print('  [pth] Could not detect model file — skipping.')
        return None, None

    sys.path.insert(0, exp_dir)
    model_mod = importlib.import_module(model_name)
    sys.path.pop(0)

    num_part = 50
    classifier = model_mod.get_model(num_part, normal_channel=args.use_normals)
    classifier.eval()
    ckpt = torch.load(os.path.join(exp_dir, 'checkpoints', 'best_model.pth'),
                      map_location='cpu', weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'])

    loader = _partseg_loader(args)
    all_pred, all_tgt = [], []

    def to_cat(y, n=16):
        return torch.eye(n)[y.cpu().data.numpy()]

    with torch.no_grad():
        for points, label, target in tqdm(loader, desc='  [pth] partseg', leave=False):
            points = points.transpose(2, 1).float()
            cat_label = to_cat(label.squeeze())
            seg_pred, _ = classifier(points, cat_label)
            pred_np = seg_pred.cpu().numpy()
            tgt_np = target.cpu().numpy()
            for i in range(points.size(0)):
                cat = SEG_LABEL_TO_CAT[tgt_np[i, 0]]
                pred_i = np.argmax(pred_np[i][:, SEG_CLASSES[cat]], axis=1) + SEG_CLASSES[cat][0]
                all_pred.append(pred_i)
                all_tgt.append(tgt_np[i])

    return _partseg_metrics(all_pred, all_tgt)


def test_partseg_onnx(args, exp_dir):
    onnx_path = find_onnx(exp_dir)
    if onnx_path is None:
        print('  [onnx] No .onnx file found — skipping.')
        return None, None

    session = load_onnx_session(onnx_path)
    input_name = session.get_inputs()[0].name
    loader = _partseg_loader(args)

    all_pred, all_tgt = [], []
    for points, label, target in tqdm(loader, desc='  [onnx] partseg', leave=False):
        pts = points.numpy().transpose(0, 2, 1).astype(np.float32)
        outputs = session.run(None, {input_name: pts})
        pred_np = outputs[0]
        tgt_np = target.numpy()
        for i in range(pts.shape[0]):
            cat = SEG_LABEL_TO_CAT[tgt_np[i, 0]]
            pred_i = np.argmax(pred_np[i][:, SEG_CLASSES[cat]], axis=1) + SEG_CLASSES[cat][0]
            all_pred.append(pred_i)
            all_tgt.append(tgt_np[i])

    return _partseg_metrics(all_pred, all_tgt)


def run_part_seg(args, exp_dir):
    header(f'Part Segmentation  |  {os.path.basename(exp_dir)}  |  N{args.num_point}')

    pth_acc = pth_iou = onnx_acc = onnx_iou = None

    if not args.skip_pth:
        pth_acc, pth_iou = test_partseg_pth(args, exp_dir)
        if pth_acc is not None:
            result_line('[pth]', pth_acc, f'mIoU: {pth_iou*100:.2f}%')

    if not args.skip_onnx:
        onnx_acc, onnx_iou = test_partseg_onnx(args, exp_dir)
        if onnx_acc is not None:
            result_line('[onnx]', onnx_acc, f'mIoU: {onnx_iou*100:.2f}%')

    if pth_acc is not None and onnx_acc is not None:
        print(f'\n  pth vs onnx: acc {abs(pth_acc-onnx_acc)*100:.2f}pp  '
              f'iou {abs(pth_iou-onnx_iou)*100:.2f}pp')


# ─────────────────────────────────────────────────────────────────────────────
# Semantic segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _semseg_loader(args):
    from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
    root = os.path.join(ROOT_DIR, 'data', 's3dis', 'stanford_indoor3d')
    ds = ScannetDatasetWholeScene(root, split='test',
                                  test_area=args.test_area,
                                  block_points=args.num_point)
    return ds


def _semseg_run_pth(classifier, dataset, args):
    num_classes = args.num_classes
    total_correct = np.zeros(num_classes)
    total_seen = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    for idx in tqdm(range(len(dataset)), desc='  [pth] semseg', leave=False):
        scene_data, scene_label, scene_smpw, scene_point_index = dataset[idx]
        whole_label = dataset.semantic_labels_list[idx]
        vote_pool = np.zeros((whole_label.shape[0], num_classes))
        num_blocks = scene_data.shape[0]
        bs = args.batch_size

        for sbatch in range((num_blocks + bs - 1) // bs):
            s, e = sbatch * bs, min((sbatch + 1) * bs, num_blocks)
            batch = np.zeros((bs, args.num_point, 9))
            batch[:e-s] = scene_data[s:e]
            t_data = torch.Tensor(batch).transpose(2, 1)
            with torch.no_grad():
                pred, _ = classifier(t_data)
            pred_np = pred.cpu().numpy()[:e-s]
            for b in range(e - s):
                for n in range(args.num_point):
                    w = scene_smpw[s+b, n]
                    if w != 0 and not np.isinf(w):
                        vote_pool[int(scene_point_index[s+b, n]), :] += pred_np[b, n]

        pred_label = np.argmax(vote_pool, axis=1)
        for l in range(num_classes):
            total_correct[l] += np.sum((pred_label == l) & (whole_label == l))
            total_seen[l] += np.sum(whole_label == l)
            total_union[l] += np.sum((pred_label == l) | (whole_label == l))

    iou = total_correct / (total_union + 1e-6)
    acc = total_correct.sum() / (total_seen.sum() + 1e-6)
    return acc, np.mean(iou)


def _semseg_run_onnx(session, dataset, args):
    input_name = session.get_inputs()[0].name
    num_classes = args.num_classes
    total_correct = np.zeros(num_classes)
    total_seen = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    for idx in tqdm(range(len(dataset)), desc='  [onnx] semseg', leave=False):
        scene_data, scene_label, scene_smpw, scene_point_index = dataset[idx]
        whole_label = dataset.semantic_labels_list[idx]
        vote_pool = np.zeros((whole_label.shape[0], num_classes))
        num_blocks = scene_data.shape[0]
        bs = args.batch_size

        for sbatch in range((num_blocks + bs - 1) // bs):
            s, e = sbatch * bs, min((sbatch + 1) * bs, num_blocks)
            batch = np.zeros((bs, args.num_point, 9), dtype=np.float32)
            batch[:e-s] = scene_data[s:e]
            batch_t = batch.transpose(0, 2, 1)
            pred_np = session.run(None, {input_name: batch_t})[0][:e-s]
            for b in range(e - s):
                for n in range(args.num_point):
                    w = scene_smpw[s+b, n]
                    if w != 0 and not np.isinf(w):
                        vote_pool[int(scene_point_index[s+b, n]), :] += pred_np[b, n]

        pred_label = np.argmax(vote_pool, axis=1)
        for l in range(num_classes):
            total_correct[l] += np.sum((pred_label == l) & (whole_label == l))
            total_seen[l] += np.sum(whole_label == l)
            total_union[l] += np.sum((pred_label == l) | (whole_label == l))

    iou = total_correct / (total_union + 1e-6)
    acc = total_correct.sum() / (total_seen.sum() + 1e-6)
    return acc, np.mean(iou)


def test_semseg_pth(args, exp_dir):
    model_name = find_model_name(exp_dir)
    if model_name is None:
        print('  [pth] Could not detect model file — skipping.')
        return None, None

    sys.path.insert(0, exp_dir)
    model_mod = importlib.import_module(model_name)
    sys.path.pop(0)

    classifier = model_mod.get_model(args.num_classes)
    classifier.eval()
    ckpt = torch.load(os.path.join(exp_dir, 'checkpoints', 'best_model.pth'),
                      map_location='cpu', weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'])

    dataset = _semseg_loader(args)
    return _semseg_run_pth(classifier, dataset, args)


def test_semseg_onnx(args, exp_dir):
    onnx_path = find_onnx(exp_dir)
    if onnx_path is None:
        print('  [onnx] No .onnx file found — skipping.')
        return None, None

    session = load_onnx_session(onnx_path)
    dataset = _semseg_loader(args)
    return _semseg_run_onnx(session, dataset, args)


def run_sem_seg(args, exp_dir):
    header(f'Semantic Segmentation  |  {os.path.basename(exp_dir)}  '
           f'|  C{args.num_classes}  N{args.num_point}  Area{args.test_area}')

    pth_acc = pth_iou = onnx_acc = onnx_iou = None

    if not args.skip_pth:
        pth_acc, pth_iou = test_semseg_pth(args, exp_dir)
        if pth_acc is not None:
            result_line('[pth]', pth_acc, f'mIoU: {pth_iou*100:.2f}%')

    if not args.skip_onnx:
        onnx_acc, onnx_iou = test_semseg_onnx(args, exp_dir)
        if onnx_acc is not None:
            result_line('[onnx]', onnx_acc, f'mIoU: {onnx_iou*100:.2f}%')

    if pth_acc is not None and onnx_acc is not None:
        print(f'\n  pth vs onnx: acc {abs(pth_acc-onnx_acc)*100:.2f}pp  '
              f'iou {abs(pth_iou-onnx_iou)*100:.2f}pp')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('Unified PointNet test runner')

    p.add_argument('--task', required=True,
                   choices=['classification', 'part_seg', 'sem_seg'],
                   help='Task type')
    p.add_argument('--log_dir', required=True,
                   help='Experiment dir name inside log/<task>/ (e.g. pointnet_10)')

    # Shared
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--use_normals', action='store_true', default=False)
    p.add_argument('--skip_pth',  action='store_true', default=False,
                   help='Skip PyTorch .pth evaluation')
    p.add_argument('--skip_onnx', action='store_true', default=False,
                   help='Skip ONNX evaluation')

    # Classification
    p.add_argument('--num_category', type=int, default=40, choices=[10, 40])
    p.add_argument('--num_point', type=int, default=1024)

    # Semantic segmentation
    p.add_argument('--num_classes', type=int, default=13)
    p.add_argument('--test_area', type=int, default=5,
                   help='S3DIS test area (1–6)')

    return p.parse_args()


def main():
    args = parse_args()

    task_to_subdir = {
        'classification': 'classification',
        'part_seg':       'part_seg',
        'sem_seg':        'sem_seg',
    }
    exp_dir = os.path.join(ROOT_DIR, 'log', task_to_subdir[args.task], args.log_dir)

    if not os.path.isdir(exp_dir):
        print(f'Error: experiment directory not found: {exp_dir}')
        sys.exit(1)

    if args.task == 'classification':
        run_classification(args, exp_dir)
    elif args.task == 'part_seg':
        run_part_seg(args, exp_dir)
    elif args.task == 'sem_seg':
        run_sem_seg(args, exp_dir)

    print()


if __name__ == '__main__':
    main()