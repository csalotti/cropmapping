# Machin Learning

This module handle everything from model definition to training/validation loops

## Structure

```
.
├── README.md           # You are here (Holà !)
├── autoencoders.py     # Self supervision model
├── classifiers.py      # Finetuning model
├── decoders.py         # Classification or reconstruct layers
├── embeddings          # Bands and positions encoding
│   ├── bands.py
│   └── position.py
├── encoders.py         # Encoders wrappers
├── gp_block.py         # Exchanger custom blocks
└── losses.py           # Focal los
```

## Elements

The training is handled through two main elements :

- `LightningModule` : Abstraction object that provides methods to define how
  training and validation is done at every steps. The gradient accumulation,
  batching and loop is handled by the super class `LightningModule` and works in
  conjonction with the `LightningDataModule` defined in [`data`](../data/).
  Current implementations are a generic [`Classifiere`](./classifiers.py) for
  finetuning and a generic [`AutoEncoder`](./autoencoders.py) for self
  supervision.

- `Layers` : Child of `torch.nn.Module`, they are the building blocks
  constituing the models on the `LightningModule`. They are provided as
  parameters on the module for a better flexibility. The current implementation
  have several categories.

  - `encoders` : Encode tha data provided by the data modul, usually using
    attention with a `TransformerEncoder`. Before being fed to the transformer,
    embeddinigs for positions and bands are applied and defined in `embeddings`
    module.

  - `decoders` : Decode the data into the final expected shape for the task. In
    the case of a classifier, it will be a classification head. For an
    autoencoders, it will be a on or more Linear Layers to retrieve the original
    information.

## Notes

- A significant amount of data is logged automatically and available on folder
  `lightning_logs` at the root.
- All logged information can be vizualised on Tensorboard. You can start a
  server by simply doing `tensorboard --logdir lightning_logs`.
- `pytorch lightning` produce checkpoint automatically that can be used to
  restart training or extract weights. More information on their
  [documentation](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
- All training have an extensive number of parameters that can be retrieved on
  the logs folder. For efficiency and flexibility, models are not pickled, but
  their parameters are
- Except for confusion matrix, distribution and metrics, all othere plots and
  save are done on the validation set.
- Due to the added computation overhead, the attention are logged every 10
  epochs, starting at 1 (1, 11, 21, ..)
- Same goes for embeddings, that can be vizualise on projector tensorboard, but
  need to be spherized beforehand.

## Configuration

Here is an example of configuration for a clasifier

```
class_path: cmap.ml.classifiers.classifier
init_args:
  encoder:
    class_path : cmap.ml.encoders.transformerencoder
    init_args:
      position_encoder:
        class_path: cmap.ml.embeddings.position.positionalencoding
        init_args:
          d_model: 256
          tau: 10000
          max_len: 10000
      bands_encoder:
        class_path: cmap.ml.embeddings.bands.pixelencoding
        init_args:
          sizes:
            - 9
            - 32
            - 64
            - 128
            - 256
      d_model: 256
      n_att_head: 8
      n_att_layers: 4
      dropout_p: 0.1
  decoder:
    class_path: cmap.ml.decoders.multiclassclassification
    init_args:
      hidden: 256
      num_classes : 7
  min_lr : 0.00001
  max_lr : 0.0001
  classes_weights:
    other : 0.52
    ble_tendre : 0.52
    mais : 0.66
    orge : 1.12
    colza : 2.64
    tournesol : 3.25
    ble_dur : 13.00
```
