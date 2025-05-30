#!/bin/sh

echo "training from scratch via train_RLMIL.py"
for STAGE in 1 2; do
  python ../train_RLMIL.py \
    --dataset CAMELYON16 \
    --data_csv path/to/data_csv.csv \
    --data_split_json path/to/data_split_json.json \
    --train_data train \
    --feat_size 1024 \
    --preload \
    --train_method scratch \
    --train_stage ${STAGE} \
    --T 6 \
    --scheduler CosineAnnealingLR \
    --batch_size 1 \
    --epochs 40 \
    --backbone_lr 0.0001 \
    --fc_lr 0.00005 \
    --arch CLAM_SB \
    --device 3 \
    --save_model \
    --exist_ok

done
python ../train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv path/to/data_csv.csv \
  --data_split_json path/to/data_split_json.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method scratch \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 1 \
  --epochs 40 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --arch CLAM_SB \
  --device 3 \
  --save_model \
  --exist_ok
