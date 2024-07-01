# config.py
import os

# Paths
lr_dir="/home/rishabh.mondal/data_download_experiments/sentinel_data/across_time/lucknow_sarath_grid_obb_v3_2022/images"
hr_dir="/home/patel_zeel/kilns_neurips24/processed_data/lucknow_sarath_grid_obb_v3/images"
val_lr="/home/rishabh.mondal/data_download_experiments/sentinel_data/across_time/lucknow_sarath_grid_obb_v3_2017/images"
val_hr="/home/patel_zeel/kilns_neurips24/processed_data/lucknow_sarath_grid_obb_v3/images"
LOG_FILE = "lucknow_sarath_training_epochs_150.log"
# Hyperparameters
lr_size = (120, 120)
hr_size = (480, 480)
batch_size = 16
learning_rate = 1e-4
num_epochs = 100
