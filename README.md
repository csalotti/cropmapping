# Crop Mapping

Fields classification from Satelite Image Time Series (SITS)

## Description

This project is an exploration to leverage and better understand transformer,
and more generally attention, in the task if classifying crops fileds. This
project leverage existing papers implementations and custom definitions,
features and processing.

## How to install ?

The proejct use python3.10 and all dependencies are specified through
[`poetry`](https://python-poetry.org/). To install all dependencies :

```bash
poetry install
```

For dependencies only useful for running the project, add option
`--without dev`.\
If you prefer another package manager, all dependencies can be exported as
`requirements.txt`

```bash
poetry export -f requirements.txt --output requirements.txt
```

## How to run ?

To run the project interactively, activate the environment with `poetry shell`
and run the following command:

```bash
python -m cmap.main fit --config configs/runs/clf_fra_19_21.yml
```

It can also be run without an interactivee shell as

```bash
poetry run python -m cmap.main fit --config configs/runs/clf_fra_19_21.yml
```

For more flexibility, `Trainer`, `DataModule` and `Model` configs are also
available separatelly and can be run as

```bash
python -m cmap.main --trainer configs/runs/finetuning.yml --model configs/runs/classifier.yml --data configs/runs/fra_19_21.yml
```

For more informations on CLI options to run the project, you can refer to
[`pytorch-lightning CLI documentation`](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_2.html)

For more informations about configurations on projects elements, check README on
modules `data` and `ml`

## Structure

```
.
├── README.md       # You're here (hello :wave:")
├── cmap            # python project
├── configs         # RPG codes, runs config and docker files
├── poetry.lock     # Poetry lock filee, source of truth
└── pyproject.toml  # Poetry dependencies definitions
```
