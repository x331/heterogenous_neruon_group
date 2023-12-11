CUDA_VISIBLE_DEVICES=0 python train.py  --lr .1 --local_module_num 16 --train_type joint --loss_type class --train_total_epochs 200 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 10  --aux_net_config 1c2f  --lsm --lsm_batch_size 770 
CUDA_VISIBLE_DEVICES=0 python train.py  --lr .1 --local_module_num 16 --train_type local --loss_type class --train_total_epochs 200 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 10  --aux_net_config 1c2f --ixx_1 5 --ixy_1 1    --ixx_2 0   --ixy_2 0  --lsm --lsm_batch_size 770 


