#!/bin/sh

echo "fine-tune via train_RLMIL.py"
for STAGE in 1 2; do
  python MuRCL/train_RLMIL.py \
    --dataset CAMELYON16 \
    --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/murcl-input_10.csv \
    --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/data_splits.json \
    --train_data train \
    --feat_size 1024 \
    --preload \
    --train_method finetune \
    --train_stage ${STAGE} \
    --checkpoint_pretrained /projects/0/prjs1477/SG-MuRCL/results/CAMELYON16_np_1024/MuRCL/T6_pd128_as0.5_pg0.1_tau1.0_alpha0.9/ABMIL/L512_D128_dpt0.0/exp/seed985/stage_3/model_best.pth.tar \
    --T 6 \
    --scheduler CosineAnnealingLR \
    --batch_size 1 \
    --epochs 40 \
    --backbone_lr 0.0001 \
    --fc_lr 0.00005 \
    --arch ABMIL \
    --device 0,1,2,3 \
    --save_model \
    --exist_ok
done
python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/murcl-input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/data_splits.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method finetune \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 1 \
  --epochs 40 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --save_model \
  --exist_ok