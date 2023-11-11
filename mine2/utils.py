import torch

#input a list of tensors with each tensor representing an exit loss
def early_exit_joint_loss(losses,wieghts=0):
    if wieghts==0:
        return torch.mean(torch.vstack(losses)).item()
    if wieghts==1:
        return torch.mean(torch.vstack(losses)).item()
        
    
    