#!/bin/sh

echo "============================="
echo "MuRCL Pretraining: Stage 1"
echo "Training MIL Aggregator M(·) and Projection Head f(·)"
echo "============================="

python MuRCL/train_MuRCL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/traintest_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/split.json \
  --feat_size 1024 \
  --preload \
  --train_stage 1 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 128 \
  --epochs 100 \
  --backbone_lr 0.0001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --exist_ok

echo "============================="
echo "MuRCL Pretraining: Stage 2"
echo "Training Reinforcement Learning Agent R"
echo "============================="

python MuRCL/train_MuRCL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/traintest_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/split.json \
  --feat_size 1024 \
  --preload \
  --train_stage 2 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 128 \
  --epochs 30 \
  --backbone_lr 0.00001 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --exist_ok

echo "============================="
echo "MuRCL Pretraining: Stage 3"
echo "Joint Fine-Tuning of M(·) and f(·)"
echo "============================="

python MuRCL/train_MuRCL.py \
  --dataset CAMELYON16 \
  --data_csv /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/traintest_input_10.csv \
  --data_split_json /projects/0/prjs1477/SG-MuRCL/data/CAMELYON16-MuRCL/split.json \
  --feat_size 1024 \
  --preload \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 128 \
  --epochs 100 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --wdecay 1e-5 \
  --patience 10 \
  --arch ABMIL \
  --device 0,1,2,3 \
  --exist_ok

echo "============================="
echo "MuRCL Pretraining Completed"
echo "============================="
