import torch

#input a list of tensors with each tensor representing an exit loss
def early_exit_joint_loss(losses,wieghts=0):
    if wieghts==0:
        return torch.mean(torch.stack(losses))
    if wieghts==1:
        return torch.mean(torch.stack(losses))
        
    

def freeze_modules_before(model, target_module):
    if target_module > 0:
        print("Starting gradient freeze")
        curr_module = 0
        for stage_i in (1, 2, 3):
            for layer_i in range(model.layers[stage_i - 1]):
                # freeze layer
                eval('model.layer' + str(stage_i))[layer_i].requires_grad = False
                
                # reached end of module
                if model.infopro_config[curr_module][0] == stage_i \
                    and model.infopro_config[curr_module][1] == layer_i:
                    print("Froze module " + str(curr_module))
                    eval('model.aux_classifier_' + str(stage_i) + '_' + str(layer_i)).requires_grad = False
                    curr_module += 1
                    if curr_module == target_module:
                        print("Finished gradient freeze")
                        return

        print("Froze everything")
        