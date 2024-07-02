name=lucknow_sarath_grid
task=obb # two options: obb, aa
yolo_task=obb # two options: obb, detect
suffix=v3
model=yolov8x-obb # yolov8x-worldv2, yolov8l-obb
epochs=100
data_folder=$name\_$task\_$suffix
data_path=/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/super_res_exp/Time-Series-Super-Resolution/crossval_sr/$data_folder
experimentName=$data_folder\_model_$model\_epochs_$epochs

echo "Name: $name"
echo "Task: $task"
echo "Suffix: $suffix"
echo "Epochs: $epochs"
echo "Data Folder: $data_folder"
echo "Data Path: $data_path"
echo "Experiment Name: $experimentName"

for fold in {0..3}
do
# fold=3
device=$fold
nohup yolo $yolo_task train model=$model.pt data=$data_path/$fold/data.yml device=$device imgsz=1120 epochs=$epochs val=False cache=True name=$experimentName\_$fold save=True save_conf=True save_txt=True > $experimentName\_$fold.log 2>&1 &
echo "Fold $fold fired on GPU $device!"
done