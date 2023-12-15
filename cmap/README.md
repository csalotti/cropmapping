# CMap

`cmap` is based of
[`pytorch-lightning`](https://lightning.ai/docs/pytorch/stable/) that is based
on 3 modules:

1. `ml` : All operations, transformations and loops for training and validation.
   It defines the `LightningModule`, asystem to organize pytorch code into
   sections (trainig ,validation, optoimizers, learning rates, ..) b. `Trainer`

## Structure

```bash
cmap
├── __init__.py
├── cli.py
├── data                    # Dataset preparation and processing
│   ├── README.md
│   ├── __init__.py
│   ├── dataset.py
│   ├── modules.py
│   └── transforms.py
├── main.py
├── ml                      # Model definition , train/val loop
│   ├── __init__.py
│   ├── autoencoders.py
│   ├── classifiers.py
│   ├── decoders.py
│   ├── embeddings
│   │   ├── __init__.py
│   │   ├── bands.py
│   │   └── position.py
│   ├── encoders.py
│   ├── gp_block.py
│   └── losses.py
└── utils                   # Plots, data conversiions and helpers
    ├── __init__.py
    ├── attention.py
    ├── chunk.py
    ├── constants.py
    ├── distributions.py
    ├── merge.py
    ├── ndvi.py
    ├── postgres.py
    └── temps.py
```
