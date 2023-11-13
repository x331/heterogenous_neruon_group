CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16  --joint_train --local_loss_mode cross_entropy --train_total_epochs 2000 --aux_net_widen 1 --aux_net_feature_dim 128  --print-freq 10 --aux_net_config 1c2f

CUDA_VISIBLE_DEVICES=0 python /train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr  --local_module_num 16 --layerwise_train --local_loss_mode cross_entropy --train_total_epochs 3200     --aux_net_widen 1 --aux_net_feature_dim 128  --print-freq 10  --aux_net_config 1c2f



CUDA_VISIBLE_DEVICES=0 python train.py  --local_module_num 8 --locally_train --infopro_classification_loss_train --infopro_classification_ratio .5 --train_total_epochs 25 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 1  --aux_net_config 1c2f --ixx_1 5 --ixy_1 1    --ixx_2 0   --ixy_2 0



CUDA_VISIBLE_DEVICES=0 python train.py  --local_module_num 10 --joint_train --infopro_classification_loss_train --infopro_classification_ratio .5 --train_total_epochs 1500 --dataset cifar10 --model resnet --layers 20 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 1  --aux_net_config 1c2f --ixx_1 5 --ixy_1 1    --ixx_2 0   --ixy_2 0

#jerry run this 5
CUDA_VISIBLE_DEVICES=0 python train.py  --local_module_num 16 --joint_train --classification_loss_train --train_total_epochs 1600 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 1  --aux_net_config 1c2f 
CUDA_VISIBLE_DEVICES=0 python train.py  --local_module_num 16 --layerwise_train --classification_loss_train --train_total_epochs 1600 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 1  --aux_net_config 1c2f 
CUDA_VISIBLE_DEVICES=0 python train.py  --local_module_num 16 --locally_train --classification_loss_train --train_total_epochs 1600 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 1  --aux_net_config 1c2f --ixx_1 5 --ixy_1 1    --ixx_2 0   --ixy_2 0
CUDA_VISIBLE_DEVICES=0 python train.py  --local_module_num 16 --locally_train --infopro_classification_loss_train --infopro_classification_ratio .5 --train_total_epochs 1600 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 1  --aux_net_config 1c2f --ixx_1 5 --ixy_1 1    --ixx_2 0   --ixy_2 0
CUDA_VISIBLE_DEVICES=0 python train.py  --local_module_num 16 --locally_train --infopro_loss_train --train_total_epochs 1600 --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr   --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --print-freq 1  --aux_net_config 1c2f --ixx_1 5 --ixy_1 1    --ixx_2 0   --ixy_2 0

