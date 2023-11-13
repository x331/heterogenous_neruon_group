Set your WANDB_API_KEY environment variable before running:

in code we have
```
if not args.no_log:
    # Ensure that the 'WANDB_API_KEY' environment variable is set in your system.
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key is None:
        raise ValueError("Please set the WANDB_API_KEY environment variable.")
    
    wandb.login(key=wandb_api_key)
    wandb.init(project='DGL-splits-resnet', entity='ghotifish', name=exp_name)
    config = wandb.config
    config.args = args
```

Before you run your script, make sure to set the environment variable WANDB_API_KEY to your actual API key. You can set an environment variable in your terminal like this:

For Linux or macOS:
```export WANDB_API_KEY='your_actual_api_key_here'```

# Layerwise Training Stuff
Enter total epochs, epoch per module will be evenly split up. For example, 50 total epochs means that each module will be ran for 25 epochs.
Currently have it so that the adjust learning rate thing resets for each module
Right now getting weird behavior from layerwise training, but might just be because hyperparameters are not calibrated for that kind of training?
