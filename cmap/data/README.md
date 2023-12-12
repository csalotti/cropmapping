## Data

This module handles every aspect related to data preparation, transformation and
iteration during training.

### Structure

```bash
├── __init__.py
├── _old                # LEGACY : old datamodules and dataset for CSV
│   ├── chunk.py
│   └── modules.py
├── dataset.py          # Iterable Dataset definition
├── modules.py          # Lightning Data Module definition
└── transforms.py       # Features and labels transforma
```

### Elements

The data module is composed of two main elements :

- `SISTSDataModule` : Based on pytorch-lightning
  [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) it
  encapsulate necessary steps to prepare and process data.
- `SITSDataset` : An
  [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)
  for pytorch that select a single point season per iteration.

It expects a dataset with the following structure:

```bash
root
 ├── train
 │    ├── features.pq
 │    ├── labels.pq
 │    └── <other_features>.pq
 └── val
      ├── features.pq
      ├── labels.pq
              └── <other_features>.pq
```

Notes:

- `SITSDataModule` and `SITSDataset` use
  [parquet file format](https://www.databricks.com/glossary/what-is-parquet#:~:text=What%20is%20Parquet%3F,handle%20complex%20data%20in%20bulk.)
  for features and labels. Features are filtered before being fetch from file
  (see [`dataset`](./dataset.py) implementation) and labels are loaded and
  sampled (see ['module'](./modules.py) and [`transforms`](./transforms.py)
- Seasons are flexible and depend on the user inputs. We defined a season as
  start in november of the preevious year and finishing the first day of
  december. It correspond to the life cycle of plants in Europe.
- The data used are points sampled from ESA Sentinel 2 acquisions. It includees
  bands sampled at 20M and a single point per field is extracted.
- In addition to the Sentinel 2 bands, it is possible to external features, such
  as temperatures and precipitations, by adding a single parquet to train and
  val folder.

### Transformations

The following elements are extracted from the time serie:

- `ts` : Sentinel 2 bands, normalized
- `days` : Days from a starting month referencee
- `mask` : Days valid (`1`) and not considered (`0`) to handle varying sequences
  sizes.
- `positions` : Elements to consider for positional encoding. It can be days,
  temperatures and even precipitations deepending on provided data

All elements are padded with respect to a number maximal of positions that is
configurable with season length and step size

### Configuration

Here is a example a Configuration used for CLI runs

```yaml
data:
  class_path: cmap.data.modules.SITSDataModule
  init_args:
  root: /mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_19_21
  rpg_mapping: configs/rpg_codes.yml
  num_workers: 4
  batch_size: 512
  train_seasons:
    - 2019
    - 2020
  val_seasons:
    - 2021
  classes:
    - other
    - ble_dur
    - ble_tendre
    - mais
    - orge
    - colza
    - tournesol
```
