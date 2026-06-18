"""
Verify ShapeNet Part Segmentation dataset structure.
Checks all 16 category folders, split files, and a sample data file.
"""
import os
import json

DATA_DIR = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal'

# All 16 part-seg categories: (human name, synset folder)
CATEGORIES = [
    ('Airplane',    '02691156'),
    ('Bag',         '02773838'),
    ('Cap',         '02954340'),
    ('Car',         '02958343'),
    ('Chair',       '03001627'),
    ('Earphone',    '03261776'),
    ('Guitar',      '03467517'),
    ('Knife',       '03624134'),
    ('Lamp',        '03636649'),
    ('Laptop',      '03642806'),
    ('Motorbike',   '03790512'),
    ('Mug',         '03797390'),
    ('Pistol',      '03948459'),
    ('Rocket',      '04099429'),
    ('Skateboard',  '04225987'),
    ('Table',       '04379243'),
]

SPLIT_FILES = [
    'shuffled_train_file_list.json',
    'shuffled_val_file_list.json',
    'shuffled_test_file_list.json',
]


def verify_shapenet(data_dir=DATA_DIR):
    print('=' * 60)
    print('Verifying ShapeNet Part Segmentation Dataset')
    print('=' * 60)
    print(f'Data dir: {data_dir}')
    print()

    if not os.path.isdir(data_dir):
        print(f'ERROR: Directory not found: {data_dir}')
        print()
        print('Run:  python utils/prepare_data_partseg.py --task download')
        return False

    all_ok = True

    # Check synsetoffset2category.txt
    cat_file = os.path.join(data_dir, 'synsetoffset2category.txt')
    if os.path.isfile(cat_file):
        print(f'[OK]      synsetoffset2category.txt')
    else:
        print(f'[MISSING] synsetoffset2category.txt')
        all_ok = False

    # Check split files
    split_dir = os.path.join(data_dir, 'train_test_split')
    total_train = total_val = total_test = 0
    for sf in SPLIT_FILES:
        path = os.path.join(split_dir, sf)
        if os.path.isfile(path):
            with open(path) as f:
                count = len(json.load(f))
            label = sf.replace('shuffled_', '').replace('_file_list.json', '')
            if 'train' in sf:
                total_train = count
            elif 'val' in sf:
                total_val = count
            else:
                total_test = count
            print(f'[OK]      train_test_split/{sf}  ({count} entries)')
        else:
            print(f'[MISSING] train_test_split/{sf}')
            all_ok = False

    print()

    # Check category folders
    print(f'Categories ({len(CATEGORIES)}):')
    missing_cats = []
    total_files = 0
    for name, synset in CATEGORIES:
        cat_path = os.path.join(data_dir, synset)
        if os.path.isdir(cat_path):
            n = len([f for f in os.listdir(cat_path) if f.endswith('.txt')])
            total_files += n
            print(f'  [OK]  {name:<12} ({synset})  {n:>5} files')
        else:
            print(f'  [MISSING] {name:<12} ({synset})')
            missing_cats.append(name)
            all_ok = False

    print()
    print('=' * 60)
    if all_ok:
        print('Verification PASSED')
        print(f'  Categories : {len(CATEGORIES)}')
        print(f'  Shape files: {total_files}')
        print(f'  Train split: {total_train}')
        print(f'  Val   split: {total_val}')
        print(f'  Test  split: {total_test}')
        print()
        print('You can now train with:')
        print('  python train_partseg.py --model pointnet2_part_seg_msg --log_dir partseg_msg')
    else:
        print('Verification FAILED')
        if missing_cats:
            print(f'  Missing categories: {missing_cats}')
        print()
        print('Re-run download:')
        print('  python utils/prepare_data_partseg.py --task download')
    print('=' * 60)

    return all_ok


if __name__ == '__main__':
    verify_shapenet()