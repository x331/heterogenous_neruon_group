import torch

#input a list of tensors with each tensor representing an exit loss
def early_exit_joint_loss(losses,wieghts=0):
    if wieghts==0:
        return torch.mean(torch.stack(losses))
    if wieghts==1:
        return torch.mean(torch.stack(losses))
        
    
def freeze_modules(model, target_module):
    # TODO: add memory of what has been frozen to check
    last_module_seen = False
    
    if target_module > 0:
        print("Starting gradient freeze")
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "aux_classifier_{}".format(target_module + 1) in name:
                if not last_module_seen:
                    print("Found last module")
                last_module_seen = True
            elif last_module_seen:
                print("---Exiting module before " + name + "---")
                return