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

