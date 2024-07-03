# config.py
import os

# Paths
lr_dir="/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/png_data_sentinel/lucknow_sarath_grid"
hr_dir="/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/png_data/lucknow_sarath_grid"
val_lr="/home/rishabh.mondal/data_download_experiments/sentinel_data/across_time/lucknow_sarath_grid_obb_v3_2020/images"
val_hr="/home/patel_zeel/kilns_neurips24/processed_data/lucknow_sarath_grid_obb_v3/images"

# Define the paths to your test dataset
test_lr_dir = '/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/super_res_exp/Time-Series-Super-Resolution/esr_gan_brick_kilns_sr/lucknow_esr_pretrained_sr_1'
output_hr_dir = '/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/super_res_exp/Time-Series-Super-Resolution/esr_gan_brick_kilns_sr/lucknow_esr_pretrained_sr_2'
os.makedirs(output_hr_dir, exist_ok=True)


# Hyperparameters
lr_size = (120, 120)
hr_size = (480, 480)
batch_size = 24
learning_rate = 1e-4
num_epochs = 100
