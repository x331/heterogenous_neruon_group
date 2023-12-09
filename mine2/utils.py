import torch
from itertools import chain

#input a list of tensors with each tensor representing an exit loss
def early_exit_joint_loss(losses,wieghts=0):
    losses = list(chain.from_iterable(losses)) # same as losses = [loss[0] for loss in losses]
    if wieghts==0:
        return torch.mean(torch.stack(losses))
    if wieghts==1:
        return torch.mean(torch.stack(losses))
        
    
    