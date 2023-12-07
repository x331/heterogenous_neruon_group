import argparse
import os
import shutil
import time
import errno
import math
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import networks.resnet
from networks.resnet import BasicBlock
from utils  import early_exit_joint_loss

from collections import defaultdict


parser = argparse.ArgumentParser(description='InfoPro-PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset: [cifar10|stl10|svhn]')

parser.add_argument('--model', default='resnet', type=str,
                    help='resnet is supported currently')

parser.add_argument('--layers', default=0, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')


# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)


# InfoPro
parser.add_argument('--local_module_num', default=1, type=int,
                    help='number of local modules (1 refers to end-to-end training)')

parser.add_argument('--balanced_memory', dest='balanced_memory', action='store_true',
                    help='whether to split local modules with balanced GPU memory (InfoPro* in the paper)')
parser.set_defaults(balanced_memory=False)

parser.add_argument('--aux_net_config', default='1c2f', type=str,
                    help='architecture of auxiliary classifier / contrastive head '
                         '(default: 1c2f; 0c1f refers to greedy SL)'
                         '[0c1f|0c2f|1c1f|1c2f|1c3f|2c2f]')

parser.add_argument('--local_loss_mode', default='contrast', type=str,
                    help='ways to estimate the task-relevant info I(x, y)'
                         '[contrast|cross_entropy]')

parser.add_argument('--aux_net_widen', default=1.0, type=float,
                    help='widen factor of the two auxiliary nets (default: 1.0)')

parser.add_argument('--aux_net_feature_dim', default=0, type=int,
                    help='number of hidden features in auxiliary classifier / contrastive head '
                         '(default: 128)')

# The hyper-parameters \lambda_1 and \lambda_2 for 1st and (K-1)th local modules.
# Note that we assume they change linearly between these two modules.
# (The last module always uses standard end-to-end loss)
# See our paper for more details.

parser.add_argument('--ixx_1', default=0.0, type=float,)   # \lambda_1 for 1st local module
parser.add_argument('--ixy_1', default=0.0, type=float,)   # \lambda_2 for 1st local module

parser.add_argument('--ixx_2', default=0.0, type=float,)   # \lambda_1 for (K-1)th local module
parser.add_argument('--ixy_2', default=0.0, type=float,)   # \lambda_2 for (K-1)th local module

#I added
# parser.add_argument('--dataloader_workers', default=12, type=int,
#                     help='number of works dataloader should have '
#                          '(default: 12)')
parser.add_argument('--train_total_epochs', default=160, type=int,
                    help='number of epochs to train for'
                         '(default: 10)')
parser.add_argument('--no_early_exit_pred', dest='no_early_exit_pred', action='store_true',
                    help='True to give just the prediction at the end of the network no early exits')
parser.add_argument('--small_datasets', dest='small_datasets', action='store_true',
                    help='True to use only dataloaders with a few hundred cases instead of all thousands')
parser.add_argument('--no_wandb_log', action='store_true',
                    help='do not log to wandb if this is set true')
parser.add_argument('--lr_decay', default=.1, type=float,
                    help='learning rate decay factor')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay factor')
parser.add_argument('--info_class_ratio', default=.5, type=float,
                    help='given value v times infopro plus (1-v) times classifcation is now the loss')
parser.add_argument('--confidence_threshold', default=0.7, type=float,
                    help='what entropy based confidence level is needed to early exit')
parser.add_argument('--lr', default=.1, type=float,
                    help='what initial learning rate you want to use')
parser.add_argument('--loss_type', default='class', type=str,
                    help='loss_type: [class|info|both]')
parser.add_argument('--train_type', default='class', type=str,
                    help='train_type: [joint|local|layer]')
parser.add_argument('--h_split', default=-1, type=int,
                    help='horizontal group split ration config')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    help='perform a dry run of the model without actual training')
parser.set_defaults(dry_run=False)
parser.add_argument('--save-checkpoint', dest='save_checkpoint', action='store_true',
                    help='save the training checkpoints')
parser.set_defaults(save_checkpoint=False)
parser.add_argument('--print-freq', default=1, type=int,
                    help='print frequency during training (default: 1)')
parser.add_argument('--lsm', dest='lsm', action='store_true',
                    help='linear seperability measure')
parser.set_defaults(lsm=False)


args = parser.parse_args()

# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': args.train_total_epochs,
        'batch_size': 1024 if args.dataset in ['cifar10', 'svhn'] else 128,
        # 'initial_learning_rate': args.lr,
        'initial_learning_rate': args.lr,
        'changing_lr': [args.train_total_epochs//2, args.train_total_epochs//4*3],
        'lr_decay_rate': .1,
        # 'lr_decay_rate': args.lr_decay,
        'momentum': 0.9,
        'nesterov': True,
        # 'weight_decay': args.weight_decay,
        'weight_decay': 1e-4,
    }
}

exp_name = str(args.train_type) \
           + '_' + str(args.dataset) \
           + '_' + str(args.model) + str(args.layers) \
           + '_L-' + str(args.loss_type) \
           + ('_ModNum-' + str(args.local_module_num)) \
           + ('_ixx1-' + str(args.ixx_1) + '_ixy1-' + str(args.ixy_1) + '_ixx2-' + str(args.ixx_2) + '_ixy2-' + str(args.ixy_2) if args.train_type == 'local' else '') \
           + ('_InfoClassRatio-' + str(args.info_class_ratio) if args.loss_type == 'both' else '') \
           + '_Epochs-' + str(args.train_total_epochs)

# exp_name = ('InfoPro*_' if args.balanced_memory else 'InfoPro_') \
#               + str(args.dataset) \
#               + '_' + str(args.model) + str(args.layers) \
#               + '_K_' + str(args.local_module_num) \
#               + '_' + str(args.name) \
#               + '/' \
#               + 'no_' + str(args.no) \
#               + '_aux_net_config_' + str(args.aux_net_config) \
#               + '_local_loss_mode_' + str(args.local_loss_mode) \
#               + '_aux_net_widen_' + str(args.aux_net_widen) \
#               + '_aux_net_feature_dim_' + str(args.aux_net_feature_dim) \
#               + '_ixx_1_' + str(args.ixx_1) \
#               + '_ixy_1_' + str(args.ixy_1) \
#               + '_ixx_2_' + str(args.ixx_2) \
#               + '_ixy_2_' + str(args.ixy_2) \
#               + ('_cos_lr_' if args.cos_lr else '') \
#               + '_train_total_epochs_' + str(args.train_total_epochs)\
#               + '_confidence_threshold_' + str(args.confidence_threshold)\
#               + '_train_type_' + str(args.train_type) \
#               + '_loss_type_' + str(args.loss_type) \
#               + '_info_class_ratio_' + str(args.info_class_ratio)\
#               + '_h_split_' + str(args.h_split)
                 
                                            
record_path = './logs/' + exp_name

record_file = record_path + '/training_process.txt'
accuracy_file = record_path + '/accuracy_epoch.txt'
loss_file = record_path + '/loss_epoch.txt'
check_point = os.path.join(record_path, args.checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.lsm:
    activations = defaultdict(list)

    def save_activation(name, transfer_to_cpu_batch=16, GPU_memory=0):
        counter = [0]  # Use a list for mutable integer

        def hook(model, input, output):
            if model.training:
                # Detach and convert to float16 to save memory
                activation = output.detach().view(output.size(0), -1).to(torch.float16)
                # print(activation.shape)

                if activations[name] is None:
                    activations[name] = [activation]  # Store as a list of tensors
                else:
                    activations[name].append(activation)

                # Transfer to CPU every 16 batches
                counter[0] += 1
                if counter[0] % transfer_to_cpu_batch == 0:
                    activations[name] = [tensor.to('cpu', non_blocking=True) for tensor in activations[name]]
                    
        def hook_A100(model, input, output):
            if model.training:
                # Detach and convert to float16 to save memory
                activation = output.detach().view(output.size(0), -1).to(torch.float16)

                if activations[name] is None:
                    activations[name] = [activation]  # Store as a list of tensors
                else:
                    activations[name].append(activation)

        if GPU_memory < 40:
            return hook
        else:
            return hook_A100


# def save_activation(name):
#     def hook(model, input, output):
#         # Check if the model is in training mode
#         if model.training:
#             # Detach and move the output to CPU
#             activation = output.detach().cpu()
#             print(f"Saving activation of {name} with shape {activation.shape}")
#             # Handle the first batch separately
#             if activations[name] is None:
#                 activations[name] = activation
#             else:
#                 # Ensure that activations[name] is a Tensor before concatenating
#                 if isinstance(activations[name], torch.Tensor):
#                     activations[name] = torch.cat((activations[name], activation), dim=0)
#                 else:
#                     raise TypeError(f"Expected activations['{name}'] to be a Tensor, but got {type(activations[name])}")
#     return hook


def main():
    
    if args.dry_run:
        print(f"Performing a dry run of {exp_name} with limited epochs...")
        args.train_total_epochs = 5  # Set the number of epochs to 5 for dry run
        training_configurations[args.model]['epochs'] = 5  # Set the number of epochs to 5 for dry run
        
        args.no_wandb_log = True  # Do not log to wandb for dry run
        args.save_checkpoint = False  # Do not save checkpoints for dry run
        args.print_freq = 10  # Set the print frequency to 1 for dry run
    
    if not args.no_wandb_log:
        # Ensure that the 'WANDB_API_KEY' environment variable is set in your system.
        wandb_api_key = os.environ.get('')
        
        wandb.login(key=wandb_api_key)
        wandb.init(project='Project-X-Experiments', entity='ghotifish', name=exp_name)
        config = wandb.config
        config.args = args


    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []

    class_num = args.dataset in ['cifar10', 'sl10', 'svhn'] and 10 or 100

    if 'cifar' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
    else:
        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}

    if args.augment:
        if 'cifar' in args.dataset:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            image_size = 32
        elif 'stl' in args.dataset:
            transform_train = transforms.Compose(
                [transforms.RandomCrop(96, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize])
            image_size = 96
        elif 'svhn' in args.dataset:
            transform_train = transforms.Compose(
                [transforms.RandomCrop(32, padding=2),
                 transforms.ToTensor(),
                 normalize])
            image_size = 32
        else:
            raise NotImplementedError

    else:
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': False}
    
    if not args.small_datasets:
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('./data', download=True, transform=transform_train,
                                                    **kwargs_dataset_train),
            batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('./data', transform=transform_test,
                                                    **kwargs_dataset_test),
            batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('./data', download=True, transform=transform_train,
                                                    **kwargs_dataset_train),
            batch_size=training_configurations[args.model]['batch_size'], sampler=torch.utils.data.SubsetRandomSampler(np.random.randint(100, size=500)),  **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('./data', transform=transform_test,
                                                    **kwargs_dataset_test),
            batch_size=training_configurations[args.model]['batch_size'], sampler=torch.utils.data.SubsetRandomSampler(np.random.randint(100, size=500)),  **kwargs)

    # create model
    if args.model == 'resnet':
        model = eval('networks.resnet.resnet' + str(args.layers))\
            (local_module_num=args.local_module_num,
             batch_size=training_configurations[args.model]['batch_size'],
             image_size=image_size,
             balanced_memory=args.balanced_memory,
             dataset=args.dataset,
             class_num=class_num,
             wide_list=(16, 16, 32, 64),
             dropout_rate=args.droprate,
             aux_net_config=args.aux_net_config,
             local_loss_mode=args.local_loss_mode,
             aux_net_widen=args.aux_net_widen,
             aux_net_feature_dim=args.aux_net_feature_dim, 
             train_type = args.train_type,
             loss_type = args.loss_type,
             info_class_ratio= args.info_class_ratio,
             h_split = args.h_split)
    else:
        raise NotImplementedError
    
    # print('module',len([module for module in model.modules()]),len([module for module in model.modules() if not isinstance(module,nn.Sequential)]), [module for module in model.modules() if not isinstance(module,nn.Sequential)][0])

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = torch.nn.DataParallel(model).to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    if not args.no_wandb_log:
        wandb.watch(model)

    if args.train_type == 'layer':
        curr_module = -1
        epochs_per_module = training_configurations[args.model]['epochs'] // model.module.local_module_num
    
    
    def register_hooks_for_sequential(module, parent_name=''):
        for name, submodule in module.named_children():
            # Construct the full name of the submodule
            module_full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(submodule, nn.Sequential):
                # If the submodule is a Sequential container, iterate its children
                for sub_name, sub_module in submodule.named_children():
                    if isinstance(sub_module, BasicBlock):
                        # Register a hook if the child is a BasicBlock
                        block_full_name = f"{module_full_name}.{sub_name}"
                        print(f"Registering hook to: {block_full_name}")
                        sub_module.register_forward_hook(save_activation(block_full_name))

            # Recursively apply this function to all submodules
            # This is to handle nested Sequential containers
            register_hooks_for_sequential(submodule, module_full_name)

    # Call this function with your model
    if args.lsm:
        register_hooks_for_sequential(model)

        
        
    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):
        if args.train_type == 'layer':  
            adjust_learning_rate(optimizer, epoch % epochs_per_module + 1)
            if epoch % epochs_per_module == 0:
                curr_module += 1
        else:
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if args.train_type == 'layer':
            train_loss, train_loss_lst, train_prec_lst, train_exits_num, train_exits_acc, LS_dict= train(train_loader, model, optimizer, epoch, curr_module)
        else:
            train_loss, train_loss_lst, train_prec_lst,  train_exits_num, train_exits_acc, LS_dict = train(train_loader, model, optimizer, epoch)
        
        
        
        train_prec1 = train_prec_lst[-1]
        
        if not args.no_wandb_log:
            wandb.log({"Train Loss": train_loss}, step=epoch)
            wandb.log({"Train Prec@1": train_prec1}, step=epoch)
            for idx, loss in enumerate(train_loss_lst):
                wandb.log({f"Train Loss_{idx}": loss}, step=epoch)
            for idx, prec in enumerate(train_prec_lst):
                wandb.log({f"Train Prec@1_{idx}": prec}, step=epoch)
            for idx, val in enumerate(train_exits_num):
                wandb.log({f"Train Number of Exits at_{idx}": val}, step=epoch)
            for idx, val in enumerate(train_exits_acc):
                wandb.log({f"Train Prec@1 when exit at_{idx}": val}, step=epoch)
            if args.lsm:
                for key, val in enumerate(LS_dict.items()):
                    wandb.log({f"Train LS_{key}": val}, step=epoch)

        # evaluate on validation set
        val_loss, val_loss_lst, val_prec_lst,  val_exits_num, val_exits_acc = validate(val_loader, model, epoch)
        if args.train_type == 'layer':
            val_prec1 = val_prec_lst[curr_module]
        else:
            val_prec1 = val_prec_lst[-1]

        if not args.no_wandb_log:
            wandb.log({"Val Loss": val_loss}, step=epoch)
            wandb.log({"Val Prec@1": val_prec1}, step=epoch)
            for idx, loss in enumerate(val_loss_lst):
                wandb.log({f"Val Loss_{idx}": loss}, step=epoch)
            for idx, prec in enumerate(val_prec_lst):
                wandb.log({f"Val Prec@1_{idx}": prec}, step=epoch)
            for idx, val in enumerate(val_exits_num):
                wandb.log({f"Val Number of Exits at_{idx}": val}, step=epoch)
            for idx, val in enumerate(val_exits_acc):
                wandb.log({f"Val Prec@1 when exit at_{idx}": val}, step=epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        if args.save_checkpoint:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_prec1,
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
            }, is_best, checkpoint=check_point)
            print('Checkpoint saved to ' + check_point)
        print('Best accuracy: ', best_prec1)
        np.savetxt(accuracy_file, np.array(val_acc))




# def estimate_memory_usage(tensor):
#     """
#     Estimate the memory usage of a tensor in bytes.
#     """
#     # Get the number of elements in the tensor
#     num_elements = tensor.numel()

#     # Estimate the size of each element in bytes
#     element_size = tensor.element_size()

#     # Total memory usage in bytes
#     memory_usage = num_elements * element_size
#     return memory_usage

# TODO: Implement this function
if args.lsm:
    def per_layer_LS_calc(activation, target_epoch, class_num, ratio=.1, batch_size=400):
        N = activation.shape[0]
        num_samples = int(N * ratio)
        print(f'num_samples: {num_samples}')
        subset_indices = torch.randperm(N)[:num_samples]
        activation = activation[subset_indices]
        target_epoch = target_epoch[subset_indices]

        # target_epoch =target_epoch.to(device='cpu')
        LS_lst = []
        for n in range(class_num):
            A = activation[(target_epoch == n)]
            A_sum = A.to(dtype=torch.float32).sum(dim=0)
            B = activation[(target_epoch != n)].to(dtype=torch.float32)
            I = A.shape[0]
            J = B.shape[0]
            D = A.shape[1]

            m_i = torch.zeros(A[0].shape, dtype=torch.float32).to(device)
            for B_j in B:
                # print(A_sum - I * B_j)
                m_i += A_sum - I * B_j
            
            w = F.normalize(m_i, p=2, dim=0)
            m_i = m_i.unsqueeze(-1)
            w = w.unsqueeze(-1)
            # print(A.shape)
            # print(B.shape)
            print(m_i.shape)
            print(w.shape)
            print(torch.matmul(w.T, m_i))

            # L2_m = torch.norm(m_i, p=2)

            # print(L2_m)
            # w = m_i / L2_m
            LS = torch.matmul(w.T, m_i).abs()

            # Initialize the result scalar on GPU
            r = torch.tensor(0, dtype=torch.float64, device=device)
            w = (w.T)
            # Process in batches
            for start in range(0, J, batch_size):
                print(start)
                end = min(start + batch_size, J)
                B_batch = B[start:end].to(device)

                # Perform operations
                A_expanded = A.unsqueeze(1)
                B_batch_expanded = B_batch.unsqueeze(0)
                diff = A_expanded - B_batch_expanded
                del A_expanded, B_batch_expanded  # Delete tensors to free up memory
                diff_reshaped = diff.reshape(-1, D)
                del diff  # Delete tensor to free up memory
                weighted_diff = (w @ diff_reshaped.T).reshape(I, -1)
                del diff_reshaped  # Delete tensor to free up memory
                r += weighted_diff.abs().sum()
                del weighted_diff  # Delete tensor to free up memory
                print('r done')

                # Clear the GPU cache
                torch.cuda.empty_cache()
                
            LS = LS / r
            
            print(f'LS for class {n}: {LS}')
            LS_lst.append(LS)
            # print(LS_lst)
        mean = torch.mean(torch.stack(LS_lst), dim=0).item()
        print(mean)
        return mean
        
        
        return 1

def train(train_loader, model, optimizer, epoch, curr_module=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = [AverageMeter() for _ in range(model.module.local_module_num)]
    per_exit_loss_meter_a = [AverageMeter() for _ in range(model.module.local_module_num)]
    per_exit_loss_meter_b = [AverageMeter() for _ in range(model.module.local_module_num)]
    per_exit_number_of_exits_meter =  [AverageMeter() for _ in range(model.module.local_module_num)]
    per_exit_acc_when_exit_meter =  [AverageMeter() for _ in range(model.module.local_module_num)]
    class_num = len(train_loader.dataset.classes)


    train_batches_num = len(train_loader)
    
    # switch to train mode
    model.train()
    if args.lsm:
        activations.clear()
    target_lst = []
    
    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.to(device)
        target_lst.append(target)
        
        x = x.to(device)

        optimizer.zero_grad()
        output, loss = model(img=x,
                             target=target,
                             ixx_1=args.ixx_1,
                             ixy_1=args.ixy_1,
                             ixx_2=args.ixx_2,
                             ixy_2=args.ixy_2,
                             target_module=curr_module)
        
        per_exit_loss = loss
        if args.train_type == 'joint':
            loss = early_exit_joint_loss(loss)
            loss.backward()
        elif args.train_type == 'layer':
            loss = loss[curr_module][0]
        else:
            loss = loss[-1][0]
            
        
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy_all_exits(output, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        for idx, meter in enumerate(top1):
            meter.update(prec1[idx].item(), x.size(0))  
        for idx, meter in enumerate(per_exit_loss_meter_a):
            meter.update(per_exit_loss[idx][0].item(), x.size(0)) 
        if len(per_exit_loss[0]) > 1:
            for idx, meter in enumerate(per_exit_loss_meter_b):
                meter.update(per_exit_loss[idx][1].item(), x.size(0)) 

        exit_num , exit_acc = accuracy_all_exits_exit_accuracy(output, target, topk=(1,),threshold=args.confidence_threshold, )
        for idx, meter in enumerate(per_exit_number_of_exits_meter):
            meter.update(exit_num[idx].item()/x.size(0)*100, x.size(0))  
        for idx, meter in enumerate(per_exit_acc_when_exit_meter):
            meter.update(exit_acc[idx].item(), exit_num[idx].item())
            

        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1) % args.print_freq == 0 or (i+1) == train_batches_num:
            # print(discriminate_weights)
            
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1[-1]))

            print(string)
            # print(weights)
            if not args.dry_run:
                fd = open(record_file, 'a+')
                fd.write(string + '\n')
                fd.write(f'per exit a loss: {[(i,meter.value,meter.ave) for i,meter in enumerate(per_exit_loss_meter_a)]}'+ '\n')
                if  len(per_exit_loss[0]) > 1:
                    fd.write(f'per exit b loss: {[(i,meter.value,meter.ave) for i,meter in enumerate(per_exit_loss_meter_b)]}'+ '\n')
                fd.write(f'per exit prec@1: {[(i,meter.value,meter.ave) for i,meter in enumerate(top1)]}'+ '\n')
                fd.write(f'per exit number of exits: {[(i,meter.value,meter.ave) for i,meter in enumerate(per_exit_number_of_exits_meter)]}'+ '\n')
                fd.write(f'per exit when exit prec@1: {[(i,meter.value,meter.ave) for i,meter in enumerate(per_exit_acc_when_exit_meter)]}'+ '\n')
                fd.close()
                
    # After training loop
    
    LS_dict = dict()
    target_epoch = torch.cat(target_lst, dim=0)
    
    if args.lsm:
        activation_keys = list(activations.keys())
        for name in activation_keys:
            if activations[name]:
                activations[name] = [tensor.to(device, non_blocking=True) for tensor in activations[name]]
                activations[name] = torch.cat(activations[name], dim=0)
                LS_dict[name] = per_layer_LS_calc(activations[name], target_epoch, class_num)
                del activations[name]
        activations.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'activation cleared')
    # total_memory = sum(estimate_memory_usage(tensor) for tensor in activations.values())
    # print(f"Total memory usage by activations: {total_memory / (1024 ** 2):.2f} MB")
    
    # ave_top1 = [meter.ave for meter in top1]
    train_loss = losses.ave
    train_loss_lst = [meter.ave for meter in per_exit_loss_meter]
    train_prec_lst= [meter.ave for meter in top1],
    train_exits_num = [meter.ave for meter in per_exit_number_of_exits_meter],
    train_exits_acc = [meter.ave for meter in per_exit_acc_when_exit_meter]
    return train_loss, train_loss_lst, train_prec_lst, train_exits_num, train_exits_acc, LS_dict


def validate(val_loader, model, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = [AverageMeter() for _ in range(model.module.local_module_num)]
    per_exit_loss_meter = [AverageMeter() for _ in range(model.module.local_module_num)]
    per_exit_number_of_exits_meter =  [AverageMeter() for _ in range(model.module.local_module_num)]
    per_exit_acc_when_exit_meter =  [AverageMeter() for _ in range(model.module.local_module_num)]
    
    train_batches_num = len(val_loader)


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output, loss = model(img=input_var,
                                 target=target_var,
                                 no_early_exit_pred = args.no_early_exit_pred)
            
        per_exit_loss = loss
        if args.train_type == 'joint' or args.train_type == 'layer':
            loss = early_exit_joint_loss(loss)
        else:
            loss = loss[-1]

        # measure accuracy and record loss
        prec1 = accuracy_all_exits(output, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        for idx, meter in enumerate(top1):
            meter.update(prec1[idx].item(), input.size(0)) 
        for idx, meter in enumerate(per_exit_loss_meter):
            meter.update(per_exit_loss[idx].item(), input.size(0)) 
            
        exit_num , exit_acc = accuracy_all_exits_exit_accuracy(output, target, topk=(1,), threshold=args.confidence_threshold)
        for idx, meter in enumerate(per_exit_number_of_exits_meter):
            meter.update(exit_num[idx].item()/input.size(0)*100, input.size(0))  
        for idx, meter in enumerate(per_exit_acc_when_exit_meter):
            meter.update(exit_acc[idx].item(), exit_num[idx].item())
                                          
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1[-1]))
    print(string)
    if not args.dry_run:
        fd = open(record_file, 'a+')
        fd.write(string + '\n')
        fd.write(f'per exit loss: {[(i,meter.value,meter.ave) for i,meter in enumerate(per_exit_loss_meter)]}'+ '\n')
        fd.write(f'per exit prec@1: {[(i,meter.value,meter.ave) for i,meter in enumerate(top1)]}'+ '\n')
        fd.write(f'per exit number of exits: {[(i,meter.value,meter.ave) for i,meter in enumerate(per_exit_number_of_exits_meter)]}'+ '\n')
        fd.write(f'per exit when exit prec@1: {[(i,meter.value,meter.ave) for i,meter in enumerate(per_exit_acc_when_exit_meter)]}'+ '\n')
        fd.close()
        
    val_acc.append(top1[-1].ave)

    # top1[-1].ave
    return losses.ave, [meter.ave for meter in per_exit_loss_meter],[meter.ave for meter in top1], [meter.ave for meter in per_exit_number_of_exits_meter], [meter.ave for meter in per_exit_acc_when_exit_meter]



def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count if self.count >0 else 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_all_exits(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = torch.stack(output).detach()

    _, pred = output.topk(maxk, 2, True, True)
    
    pred = pred.reshape(pred.shape[0],pred.shape[2],pred.shape[1])
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    


    


    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:,:k].reshape(correct.shape[0],-1).float().sum(1)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# def accuracy_all_exits_exit_accuracy(output, target, topk=(1,),threshold=.7):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#     output = torch.stack(output).detach()

#     _, pred = output.topk(maxk, 2, True, True)
    
#     prob = torch.softmax(output,dim=2)
#     p = (1/(np.log(output.shape[2])))* (prob*torch.log(prob)).sum(dim=2,keepdim=True)
    
#     pred = pred.reshape(pred.shape[0],pred.shape[2],pred.shape[1])
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
    
#     prob = torch.softmax(output,dim=2)
#     p = ((1/(np.log(output.shape[2])))* (prob*torch.log(prob)).sum(dim=2)).cpu()
#     p = p+1
#     e = p>threshold
#     exits = torch.zeros(p.shape[0],1,device='cpu')
#     exits_acc = torch.zeros(p.shape[0],1,device='cpu')
#     for m in range(p.shape[0]):
#         print(f'intotal: {p[m].shape} before + 1:{((p >= 0) & (p[m] < 1)).sum().item()} in 0-1')
#         exits[m] = e[m].sum()
#     for m in range(p.shape[0]):
#         if exits[m] != 0:
#             exit_preds = correct[m,:1][e[m].reshape(1,e.shape[1])]
#             sum = exit_preds.float().sum()
#             avg =  sum/exit_preds.shape[0]*100.0
#             exits_acc[m] = avg

#     return exits, exits_acc

def accuracy_all_exits_exit_accuracy(output, target, topk=(1,), threshold=.7, device=device):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output = torch.stack(output).detach()

    _, pred = output.topk(maxk, 2, True, True)
    prob = torch.softmax(output, dim=2)
    # p = (1 / (np.log(output.shape[2]))) * (prob * torch.log(prob)).sum(dim=2, keepdim=True)

    pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[1])
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    prob = torch.softmax(output, dim=2)
    p = ((1 / (np.log(output.shape[2]))) * (prob * torch.log(prob)).sum(dim=2)) # p = - normalized entropy
    if device is not None:
        p = p.to(device)  # Move p to the specified device
    p = p + 1  # p = 1 - normalized entropy, p > threshold means  normalized entropy < 1 - threshold (i.e. 0.3)
    e = p > threshold 
    exits = torch.zeros(p.shape[0], 1, device=device)  # Use the specified device
    exits_acc = torch.zeros(p.shape[0], 1, device=device)  # Use the specified device

    for m in range(p.shape[0]):
        exits[m] = e[m].sum()
        if exits[m] != 0:
            exit_preds = correct[m, :1][e[m].reshape(1, e.shape[1])]
            sum = exit_preds.float().sum()
            avg = sum / exit_preds.shape[0] * 100.0
            exits_acc[m] = avg

    return exits, exits_acc


if __name__ == '__main__':
    main()
