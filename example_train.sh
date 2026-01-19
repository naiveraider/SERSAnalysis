#!/bin/bash
# 训练示例脚本

echo "训练CNN模型..."
python train.py \
    --model cnn \
    --data_dir datasets \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --model_save_path model/saved_model_cnn

echo ""
echo "训练TCN模型..."
python train.py \
    --model tcn \
    --data_dir datasets \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --tcn_num_channels 64 128 256 \
    --tcn_kernel_size 3 \
    --tcn_dropout 0.2 \
    --model_save_path model/saved_model_tcn

