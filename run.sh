python3.7 train.py --data_root /home/u20/d2/dataset/DroneRGBT/save \
    --dataset_file SHHA \
    --epochs 10 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
# python3.7 run_test.py --weight_path ./weights/best_mae.pth --output_dir ./logs/