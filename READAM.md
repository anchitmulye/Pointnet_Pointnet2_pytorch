

## PointNet

### Classification

## Common Commands

### Download ModelNet40

```bash
python utils/prepare_data.py --task download --num_category 40
```

### Convert ModelNet40

```bash
python utils/prepare_data.py --task convert --num_category 40 --source data/ModelNet40
```

### Train on CPU

```bash
python main.py --task train --num_category 40 --epoch 1 --use_cpu
```

### Test a Checkpoint on CPU


```bash
python main.py --task test \
  --checkpoint log/classification/<run_id>/checkpoints/best_model.pth \
  --log_dir <run_id> \
  --use_cpu
```

### Export Classification Model to ONNX

```bash
python onnx/export_classification.py \
  --checkpoint log/classification/<run_id>/checkpoints/best_model.pth \
  --model pointnet_cls
```

## Notes

- Use `--num_category 10` for ModelNet10 and `--num_category 40` for ModelNet40.
- Replace `<run_id>` with the timestamped directory created during training.
- Use `--use_cpu` when running without GPU support.
- Remove `--use_cpu` if your environment is configured for GPU training and the scripts support CUDA execution.

## Author

Maintained by [anchitmulye](https://github.com/anchitmulye).