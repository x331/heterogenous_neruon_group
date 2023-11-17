import torch

#input a list of tensors with each tensor representing an exit loss
def early_exit_joint_loss(losses,wieghts=0):
    if wieghts==0:
        return torch.mean(torch.stack(losses))
    if wieghts==1:
        return torch.mean(torch.stack(losses))
        
    

def freeze_module_before(model, target_module):
    if target_module > 0:
        print("Starting module freeze")
        stage = model.infopro_config[target_module - 1][0]
        # layer right after the end of the previous module
        if target_module > 1:
            start_layer = model.infopro_config[target_module - 2][1] + 1
        else:
            start_layer = 0
        
        for layer_i in range(start_layer, model.layers[stage - 1]):
            # freeze layer
            eval('model.layer' + str(stage))[layer_i].requires_grad = False
            
            # reached end of module
            if model.infopro_config[target_module - 1][1] == layer_i:
                eval('model.aux_classifier_' + str(stage) + '_' + str(layer_i)).requires_grad = False
                print("Finished module freeze")
                return

        print("Froze everything")
        