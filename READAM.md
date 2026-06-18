# PointNet / PointNet++ — Commands Reference

Maintained by [anchitmulye](https://github.com/anchitmulye).

All commands run from the `pointnet_pointnet2/` directory.  
Add `--use_cpu` to any train/test command when running without a GPU.

---

## Environment

```bash
conda env create -f env.yml
conda activate pointnet
```

---

## ModelNet — Classification

### 1. Data preparation

```bash
# Download ModelNet40 (normal-resampled, ready to use)
python utils/prepare_data.py --task download --num_category 40

# Download ModelNet10
python utils/prepare_data.py --task download --num_category 10

# Convert raw .off files if you have your own copy
python utils/prepare_data.py --task convert --num_category 40 --source data/ModelNet40
```

### 2. Train

```bash
# PointNet — ModelNet40
python train_classification.py --model pointnet_cls --num_category 40 --log_dir pointnet_40

# PointNet — ModelNet10
python train_classification.py --model pointnet_cls --num_category 10 --log_dir pointnet_10

# PointNet++ SSG — ModelNet40 (no normals)
python train_classification.py --model pointnet2_cls_ssg --num_category 40 --log_dir pointnet2_ssg_wo_normals

# PointNet++ SSG — ModelNet10
python train_classification.py --model pointnet2_cls_ssg --num_category 10 --log_dir pointnet2_ssg_10

# PointNet++ MSG — ModelNet40 (with normals)
python train_classification.py --model pointnet2_cls_msg --num_category 40 --use_normals --log_dir pointnet2_msg_normals

# CPU flag (append to any train command)
python train_classification.py --model pointnet_cls --num_category 10 --use_cpu --log_dir pointnet_10
```

### 3. Test

```bash
# PyTorch checkpoint
python test_classification.py --log_dir pointnet_10 --num_category 10 --use_cpu

# Unified pth + ONNX side-by-side
python test/test.py --task classification --log_dir pointnet_10 --num_category 10
python test/test.py --task classification --log_dir pointnet2_ssg_wo_normals --num_category 40
python test/test.py --task classification --log_dir pointnet2_msg_normals --num_category 40 --use_normals
```

### 4. Export to ONNX

```bash
# PointNet
python onnx/export_classification.py \
  --model pointnet_cls \
  --checkpoint log/classification/pointnet_10/checkpoints/best_model.pth \
  --num_category 10

# PointNet++ SSG
python onnx/export_pointnet2_cls.py \
  --model pointnet2_cls_ssg \
  --checkpoint log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth \
  --num_category 40

# PointNet++ MSG (with normals)
python onnx/export_pointnet2_cls.py \
  --model pointnet2_cls_msg \
  --checkpoint log/classification/pointnet2_msg_normals/checkpoints/best_model.pth \
  --num_category 40 --use_normals
```

---

## ShapeNet — Part Segmentation

### 1. Data preparation

```bash
# Download from Hugging Face (~709 MB) — requires huggingface_hub
python utils/prepare_data_partseg.py --task download

# OR extract a zip you downloaded from the browser
python utils/prepare_data_partseg.py --task extract \
  --zip_path "C:\Users\<you>\Downloads\shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"

# Verify structure
python utils/prepare_data_partseg.py --task check
```

### 2. Train

```bash
# PointNet++ MSG (recommended)
python train_partseg.py --model pointnet2_part_seg_msg --log_dir partseg_msg

# CPU (smaller batch + fewer points for speed)
python train_partseg.py --model pointnet2_part_seg_msg --log_dir partseg_msg \
  --use_cpu --batch_size 4 --npoint 1024 --epoch 5
```

### 3. Test

```bash
python test/test.py --task part_seg --log_dir pointnet2_part_seg_msg
```

### 4. Export to ONNX

```bash
python onnx/export_pointnet2_partseg.py \
  --checkpoint log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth
```

---

## S3DIS — Semantic Segmentation

### 1. Data preparation

S3DIS requires a manual download from Stanford (access request required):

```bash
# Read instructions
python utils/prepare_data_seg.py --task download --seg_task sem_seg

# After extracting the dataset to data/s3dis/ — convert all rooms to .npy
python utils/prepare_data_seg.py --task convert --seg_task sem_seg

# Convert a single area only
python utils/prepare_data_seg.py --task convert --seg_task sem_seg --area 5

# Check converted files
python utils/prepare_data_seg.py --task check --seg_task sem_seg
```

### 2. Train

```bash
# PointNet++ SSG
python train_semseg.py --model pointnet2_sem_seg --log_dir pointnet2_sem_seg

# PointNet
python train_semseg.py --model pointnet_sem_seg --log_dir pointnet_sem_seg
```

### 3. Test

```bash
python test/test.py --task sem_seg --log_dir pointnet_sem_seg --num_classes 13
python test/test.py --task sem_seg --log_dir pointnet2_sem_seg --num_classes 13
```

### 4. Export to ONNX

```bash
python onnx/export_pointnet2_semseg.py \
  --checkpoint log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth
```

---

## Notes

| Topic | Detail |
|---|---|
| ModelNet categories | `--num_category 10` for ModelNet10, `40` for ModelNet40 |
| Normals | Add `--use_normals` for 6-channel input (xyz + normals) |
| CPU training | Add `--use_cpu` to any train/test command |
| Log dirs | Pass the folder name inside `log/<task>/`, not the full path |
| ONNX models | Saved to `log/<task>/<log_dir>/onnx/` after export |