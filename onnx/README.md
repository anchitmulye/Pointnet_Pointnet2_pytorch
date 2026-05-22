# ONNX Commands

## Export to ONNX

### Classification
```bash
python onnx/export_onnx_classification.py --model pointnet_cls --checkpoint log/classification/test/checkpoints/best_model.pth --num_category 10 --num_point 1024
```

### Part Segmentation
```bash
python onnx/export_onnx_partseg.py --model pointnet_part_seg --checkpoint log/part_seg/pointnet_part_seg/checkpoints/best_model.pth --num_point 2048
```

### Semantic Segmentation
```bash
python onnx/export_onnx_semseg.py --model pointnet_sem_seg --checkpoint log/sem_seg/pointnet_sem_seg/checkpoints/best_model.pth --num_classes 13 --num_point 4096
```

## Test ONNX Models

### Classification
```bash
python onnx/test_onnx_classification.py --onnx_path log/classification/test/onnx/pointnet_cls_c10_n1024.onnx --num_category 10 --num_point 1024
```

### Part Segmentation
```bash
python onnx/test_onnx_partseg.py --onnx_path log/part_seg/pointnet_part_seg/onnx/pointnet_part_seg_n2048.onnx --num_point 2048
```

### Semantic Segmentation
```bash
python onnx/test_onnx_semseg.py --onnx_path log/sem_seg/pointnet_sem_seg/onnx/pointnet_sem_seg_c13_n4096.onnx --test_area 5 --num_votes 3
```

## Notes

- Only PointNet models are ONNX compatible
- PointNet++ uses custom CUDA ops and cannot be exported to ONNX right now
- Install dependencies: `pip install onnx onnxruntime`