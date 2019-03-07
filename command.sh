kitti_odometry_dir='/home/tianwei/Data/kitti/odometry/dataset/odometry/'
kitti_dir='/home/tianwei/Data/kitti/' 
kitti_raw_dir=$kitti_dir'/raw/' 
make3d_dir='/home/tianwei/Data/make3d/'
kitti_raw_dump_dir=$kitti_dir'/raw_dump/' 
kitti_eigen_test_dir=$kitti_dir'/eigen_test/' 
kitti_odom=$kitti_dir'/kitti_odom/' 
kitti_odom5=$kitti_dir'/kitti_odom5/'
kitti_odom_match3=$kitti_dir'/kitti_odom_match3/'
kitti_odom_match5=$kitti_dir'/kitti_odom_match5/'
kitti_raw_odom=$kitti_dir'/odometry/dataset/odometry/'
cityscapes_dir='/home/tianwei/Data/cityscapes'
cityscapes_dump='/home/tianwei/Data/cityscapes/dump/'
output_folder=./output/
root_folder='./'
model_idx=258000
save_freq_step=4000
checkpoint_dir=./ckpt/ 
# kitti eval depth
depth_pred_file='output/model-'$model_idx'.npy' 

# Generate training and testing data
## for odometry dataset
python data/prepare_train_data.py --dataset_dir=$kitti_raw_odom --dataset_name=kitti_odom \
    --dump_root=$kitti_odom_match3 --seq_length=3 --img_width=416 --img_height=128 \
    --num_threads=8 --generate_test True
## for raw dataset (Eigen split)
python data/prepare_train_data.py --dataset_dir=$kitti_raw_dir --dataset_name=kitti_raw_eigen \
    --dump_root=$kitti_raw_dump_dir --seq_length=3 --img_width=416 --img_height=128 \
    --num_threads=8 --match_num $match_num

# Train on KITTI odometry dataset
match_num=100
python train.py --dataset_dir=$kitti_odom_match3 --checkpoint_dir=$checkpoint_dir --img_width=416 --img_height=128 --batch_size=4 --seq_length 3 \
    --max_steps 300000 --save_freq 2000 --learning_rate 0.001 --num_scales 1 --init_ckpt_file $checkpoint_dir'model-'$model_idx --continue_train=True --match_num $match_num

# Train on KITTI Eigen split
python train.py --dataset_dir=$kitti_raw_dump_dir --checkpoint_dir=$checkpoint_dir --img_width=416 --img_height=128 --batch_size=4 --seq_length 3 \
    --max_steps 300000 --save_freq $save_freq_step --learning_rate 0.001 --num_scales 1 --match_num $match_num --init_ckpt_file $checkpoint_dir'model-'$model_idx --continue_train=True 

# Testing depth model
r=250000
depth_ckpt_file=$rootfolder$checkpoint_dir'model-'$r
depth_pred_file='output/model-'$r'.npy' 
python test_kitti_depth.py --dataset_dir $kitti_raw_dir --output_dir $output_folder --ckpt_file $depth_ckpt_file #--show
python kitti_eval/eval_depth.py --kitti_dir=$kitti_raw_dir --pred_file $depth_pred_file #--show True --use_interp_depth True

# Testing pose model
sl=3
r=258000
pose_ckpt_file=$root_folder$checkpoint_dir'model-'$r
for seq_num in 09 10
do 
    rm -rf $output_folder/$seq_num/
    echo 'seq '$seq_num
    python test_kitti_pose.py --test_seq $seq_num --dataset_dir $kitti_raw_odom --output_dir $output_folder'/'$seq_num'/' --ckpt_file $pose_ckpt_file --seq_length $sl --concat_img_dir $kitti_odom_match3
    python kitti_eval/eval_pose.py --gtruth_dir=$root_folder'kitti_eval/pose_data/ground_truth/seq'$sl'/'$seq_num/  --pred_dir=$output_folder'/'$seq_num'/'
done
