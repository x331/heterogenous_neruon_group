CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16  --joint_train --local_loss_mode cross_entropy --train_total_epochs 2000 --aux_net_widen 1 --aux_net_feature_dim 128  --print-freq 10 --aux_net_config 1c2f

CUDA_VISIBLE_DEVICES=0 python /train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr  --local_module_num 16 --layerwise_train --local_loss_mode cross_entropy --train_total_epochs 3200     --aux_net_widen 1 --aux_net_feature_dim 128  --print-freq 10  --aux_net_config 1c2f
