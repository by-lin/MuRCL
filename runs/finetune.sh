#!/bin/sh

echo "============================="
echo "Fine-Tuning Stage 1: Initializing with Pretrained Weights"
echo "============================="

python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/traintest_input.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/split.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method finetune \
  --train_stage 1 \
  --checkpoint_pretrained /projects/0/prjs1477/SG-MuRCL/path/to/results/best.tar \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 4 \
  --epochs 100 \
  --backbone_lr 0.0001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --save_model \
  --exist_ok

echo "============================="
echo "Fine-Tuning Stage 2: Training RL Agent"
echo "============================="

python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/traintest_input.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/split.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method finetune \
  --train_stage 2 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 4 \
  --epochs 30 \
  --backbone_lr 0.00001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --save_model \
  --exist_ok

echo "============================="
echo "Fine-Tuning Stage 3: Joint Optimization of MIL and RL"
echo "============================="

python MuRCL/train_RLMIL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/traintest_input.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/split.json \
  --train_data train \
  --feat_size 1024 \
  --preload \
  --train_method finetune \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 4 \
  --epochs 100 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --save_model \
  --exist_ok

echo "============================="
echo "Fine-Tuning Completed ✅"
echo "Check saved models under the specified save_dir."
echo "============================="