#!/bin/sh

echo "Stage 1 & 2: Pre-training MuRCL with ABMIL + ResNet features on 4× A100"

for STAGE in 1 2; do
  python MuRCL/train_MuRCL.py \
    --dataset CAMELYON16 \
    --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/murcl-input_10.csv \
    --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/data_splits.json \
    --feat_size 1024 \
    --preload \
    --train_stage ${STAGE} \
    --T 6 \
    --batch_size 512 \
    --epochs 100 \
    --backbone_lr 1e-4 \
    --fc_lr 1e-4 \
    --scheduler CosineAnnealingLR \
    --wdecay 1e-5 \
    --alpha 0.9 \
    --arch ABMIL \
    --model_dim 512 \
    --D 128 \
    --dropout 0.0 \
    --ppo_lr 1e-5 \
    --ppo_gamma 0.1 \
    --K_epochs 3 \
    --policy_hidden_dim 512 \
    --action_std 0.5 \
    --device 0,1,2,3 \
    --exist_ok
done

echo "Stage 3: Fine-tuning MuRCL with ABMIL"

python MuRCL/train_MuRCL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/murcl-input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-baseline/data_splits.json \
  --feat_size 1024 \
  --preload \
  --train_stage 3 \
  --T 6 \
  --batch_size 512 \
  --epochs 100 \
  --backbone_lr 5e-5 \
  --fc_lr 1e-5 \
  --scheduler StepLR \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --model_dim 512 \
  --D 128 \
  --dropout 0.0 \
  --device 0,1,2,3 \
  --exist_ok
