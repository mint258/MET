# MET: Molecular Equivariant Transformer

Repository for:

**Integrating equivariant architectures and charge supervision for data-efficient molecular property prediction**

- Journal status: accepted by *Molecular Systems Design & Engineering (MSDE)*
- DOI: https://doi.org/10.1039/D5ME00173K
- PyPI package: https://pypi.org/project/mint258-met/
- Prebuilt LMDB dataset release: https://huggingface.co/datasets/Mint258/MET-dateset
- Checkpoint release: https://huggingface.co/Mint258/MET-models

## Overview

MET first pretrains an equivariant 3D molecular encoder on **atomic partial charge prediction** and then fine-tunes the encoder on downstream molecular property prediction.

This maintained repository includes:

- an installable PyPI package for embedding extraction and downstream fine-tuning
- unified pretraining, fine-tuning, and evaluation entrypoints
- deterministic manifest generation for the low-data settings in the paper
- QM9 support in both `xyz` and `lmdb` formats for pretraining
- RDKit-based conformer generation for downstream fine-tuning from 1D inputs such as SMILES
- multi-conformer averaging during downstream training and inference
- checkpoint loading utilities for both pretrained and fine-tuned `.pth` files
- pure-PyTorch fallbacks for graph and scatter ops, so `torch_cluster` / `torch_scatter` are optional rather than mandatory
- a minimal environment file and requirements list derived from the actual imports in this codebase

## PyPI API

For direct downstream use, install the published package instead of cloning the whole repository:

```bash
pip install mint258-met
```

The PyPI package is fine-tuning-oriented. It downloads the released pretrained MET checkpoint from Hugging Face by default and exposes two high-level APIs:

- `METEncoder` for atom-level embeddings with shape `[N, D]`
- `METFineTuner` for downstream regression with built-in CSV / XYZ dataloaders

### Minimal Embedding Example

```python
from met import METEncoder

encoder = METEncoder.from_pretrained(device="cpu")
atom_embeddings = encoder.embed_smiles("CCO", num_conformers=4)
print(atom_embeddings.shape)  # [N, D]
```

You can also embed from `xyz` input:

```python
atom_embeddings = encoder.embed_xyz("molecule.xyz")
```

### Minimal Fine-Tuning Example

```python
from met import METFineTuner

model = METFineTuner.from_pretrained(
    device="cpu",
    output_dim=1,
    hidden_dim=256,
    layers=2,
    dropout=0.1,
    pooling_time="after_last_layer",
    pooling_method="attention",
    normalization="layernorm",
)

model.set_trainable_encoder_blocks(trainable_blocks=1)

history = model.fit_csv(
    csv_path="my_dataset.csv",
    smiles_column="smiles",
    target_columns=["target"],
    num_conformers=8,
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
)
```

### API Summary

`METEncoder` methods:

- `from_pretrained(...)`
- `embed_smiles(...)`
- `embed_xyz(...)`
- `embed_smiles_batch(...)`

`METFineTuner` methods:

- `from_pretrained(...)`
- `fit_csv(...)`
- `fit_xyz(...)`
- `predict_csv(...)`
- `predict_xyz(...)`
- `predict_smiles(...)`
- `evaluate_csv(...)`
- `evaluate_xyz(...)`
- `freeze_encoder()`
- `unfreeze_encoder()`
- `set_trainable_encoder_blocks(...)`
- `set_trainable_modules(...)`

### Default Downstream Head

The default fine-tuning head can be configured through:

- `output_dim`
- `hidden_dim`
- `layers`
- `dropout`
- `pooling_time`
- `pooling_method`
- `normalization`

`pooling_time="after_last_layer"` means node embeddings are transformed first and pooled at the end.  
`pooling_time="before_head"` means node embeddings are pooled first and the graph-level representation is refined afterward.

### Custom Heads

You can replace the built-in head with your own `torch.nn.Module`:

```python
import torch.nn as nn
from met import METFineTuner


class MyHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, node_embeddings, mask):
        pooled = node_embeddings.mean(dim=1)
        return self.proj(pooled)


model = METFineTuner.from_pretrained(device="cpu", output_dim=1, head=MyHead(128, 1))
```

The custom head must accept:

- `node_embeddings` with shape `[batch, max_nodes, dim]`
- `mask` with shape `[batch, max_nodes]`

and return predictions with shape `[batch, output_dim]`.

### Package Source

The package source used to build the PyPI release is stored in `PyPI/`.

## Dataset Release

Prebuilt QM7 and QM9 LMDB files are hosted on Hugging Face rather than in this GitHub repository:

https://huggingface.co/datasets/Mint258/MET-dateset

Download the LMDB files from that dataset page and place them under `data/QM7/` and `data/QM9/` locally if you want to use the ready-made binary datasets. The conversion scripts in this repository are still available if you prefer to regenerate the LMDB files from local `xyz` inputs.

## Checkpoint Release

Pretrained and fine-tuned checkpoints are hosted on Hugging Face rather than in this GitHub repository:

https://huggingface.co/Mint258/MET-models

If you want to reuse the released weights with the existing commands in this repository, download the files from that model page and place them under local `pretrained_ckpt/` and `fine-tuned_ckpt/` directories.

## Repository Layout

```text
MET/
├── data/
│   ├── QM7/
│   └── QM9/
├── pretrain/
├── fine-tune/
├── alignment_analysis/
├── scripts/
├── MET.yml
├── requirements-minimal.txt
└── requirements-optional.txt
```

## Minimal Dependencies

The minimum runtime dependencies used by training, fine-tuning, evaluation, and dataset conversion are:

- `torch`
- `torch-geometric`
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `sympy`
- `rdkit`
- `lmdb`
- `periodictable`
- `tqdm`

Optional:

- `seaborn` for `alignment_analysis/`

### Conda Setup

```bash
conda env create -f MET.yml
conda activate met
```

`MET.yml` is intentionally minimal and is not an exported workstation snapshot.

If you want GPU training, replace `cpuonly` in `MET.yml` with the CUDA variant that matches your machine.

## Data Formats

### QM9 Pretraining Input

Pretraining now accepts either:

- an `xyz` directory such as `data/QM9/train_valid_database`
- a manifest of `xyz` files such as `manifests/qm9_train_100.txt`
- a single-file `lmdb` database such as `data/QM9/train_valid_database.lmdb`

Prebuilt LMDB files are available from the Hugging Face dataset release above. If you still keep the original `xyz` files locally, you can also regenerate the LMDB files with the scripts below.

### Downstream Fine-Tuning Input

Fine-tuning remains **single-file / plain-file based** rather than LMDB-based:

- `xyz` directories or manifests for the original QM7 / QM9 downstream experiments
- a single `.csv` file, a directory of `.csv` files, or a manifest that references `.csv` files

For CSV input:

- one SMILES column is required: `smiles`, `SMILE`, `SMILES`, or `smile`
- all numeric columns except common ID/name columns are treated as targets
- RDKit is used to build 3D conformers before the molecule is passed into MET

## Property Names

### QM9

`rot_A`, `rot_B`, `rot_C`, `dipole`, `polarizability`, `HOMO_energy`, `LUMO_energy`, `gap`, `R2`, `zpve`, `U0`, `U298`, `H298`, `G298`, `Cv`

### QM7

`atomization_energy`

Backward-compatible aliases such as `P0`, `P1`, `P2`, and the legacy `rot_A` alias for the QM7 target are still accepted.

## Reproducing the Low-Data Splits

Generate deterministic manifests once:

```bash
python scripts/create_manifests.py --output_dir manifests --seed 42
```

This produces the paper-style subsets:

- `manifests/qm7_train_100.txt`
- `manifests/qm7_train_300.txt`
- `manifests/qm7_train_500.txt`
- `manifests/qm7_train_800.txt`
- `manifests/qm7_train_1000.txt`
- `manifests/qm9_train_100.txt`
- `manifests/qm9_train_1000.txt`
- `manifests/qm9_train_5000.txt`
- `manifests/qm9_train_20000.txt`
- `manifests/qm9_train_100000.txt`

## Building QM9 LMDB Files

Convert any QM9 `xyz` directory or manifest into a single-file LMDB database:

```bash
python scripts/build_qm9_lmdb.py \
  --input_root data/QM9/train_valid_database \
  --output_path data/QM9/train_valid_database.lmdb \
  --map_size_gb 0.064
```

The builder now auto-expands the LMDB map size only when needed, so the final `.lmdb` file stays close to the true dataset size instead of reserving multiple gigabytes up front.

You can also call the same workflow via:

```bash
python scripts/run_experiment.py build-qm9-lmdb
```

## Building QM7 LMDB Files

Convert any QM7 `xyz` directory or manifest into a single-file LMDB database:

```bash
python scripts/build_qm7_lmdb.py \
  --input_root data/QM7/full_database \
  --output_path data/QM7/full_database.lmdb \
  --map_size_gb 0.064
```

You can also call the same workflow via:

```bash
python scripts/run_experiment.py build-qm7-lmdb
```

## Pretraining

### Full QM9 Charge Pretraining from LMDB

```bash
python pretrain/training_charge_model.py \
  --data_root data/QM9/train_valid_database.lmdb \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --atom_embedding_dim 128 \
  --hidden_channels 256 \
  --middle_channels 256 \
  --num_layers 4 \
  --num_radial 8 \
  --num_spherical 5 \
  --transformer_layers 1 \
  --transformer_heads_z 1 \
  --device cpu \
  --save_path pretrained_ckpt/best_model_dim128_reproduced.pth
```

The same command also works with `xyz` input, for example `--data_root data/QM9/train_valid_database`.

### LMDB Smoke Test Used During This Update

```bash
python pretrain/training_charge_model.py \
  --data_root data/QM9/test_database.lmdb \
  --epochs 1 \
  --batch_size 8 \
  --device cpu \
  --save_path outputs/smoke_pretrain_lmdb/best_model.pth
```

Verified in this update:

- LMDB evaluation succeeded with `pretrained_ckpt/best_model_dim128.pth`
- LMDB pretraining smoke test completed successfully
- a readable checkpoint was produced at `outputs/smoke_pretrain_lmdb/best_model.pth`

## Fine-Tuning

### QM7 Atomization Energy

```bash
python fine-tune/fine_tune_training.py \
  --pretrained_checkpoint_path pretrained_ckpt/best_model_dim128.pth \
  --data_manifest manifests/qm7_train_300.txt \
  --target_property atomization_energy \
  --dropout 0 \
  --learning_rate 1e-4 \
  --freeze_up_to_layer 4 \
  --num_layers 0 \
  --num_heads 1 \
  --epochs 100 \
  --batch_size 32 \
  --device cpu \
  --save_model fine-tuned_ckpt/qm7/qm7_data300_reproduced.pth
```

### QM9 Dipole

```bash
python fine-tune/fine_tune_training.py \
  --pretrained_checkpoint_path pretrained_ckpt/best_model_dim128.pth \
  --data_manifest manifests/qm9_train_1000.txt \
  --target_property dipole \
  --dropout 0 \
  --learning_rate 1e-4 \
  --freeze_up_to_layer 4 \
  --num_linear_layers 0 \
  --epochs 100 \
  --batch_size 32 \
  --device cpu \
  --save_model fine-tuned_ckpt/dipole/dipole_data1000_reproduced.pth
```

### Fine-Tuning from SMILES with RDKit Conformer Generation

If your downstream dataset only contains 1D molecular strings, pass a CSV file directly.

Example CSV:

```text
smiles,toy_property
CCO,0.5
CCN,0.8
```

Example training command with multi-conformer averaging:

```bash
python fine-tune/fine_tune_training.py \
  --pretrained_checkpoint_path pretrained_ckpt/best_model_dim128.pth \
  --data_root my_smiles_dataset.csv \
  --target_property toy_property \
  --num_conformers 8 \
  --conformer_seed 2024 \
  --epochs 100 \
  --batch_size 16 \
  --device cpu \
  --save_model outputs/my_smiles_model.pth
```

Notes:

- `--num_conformers` controls how many 3D conformers RDKit generates per SMILES
- training and inference average predictions across conformers automatically
- use `--disable_uff_optimization` if you want to skip RDKit UFF cleanup after ETKDG embedding

### Smoke Tests Used During This Update

XYZ fine-tuning regression smoke test:

```bash
python fine-tune/fine_tune_training.py \
  --pretrained_checkpoint_path pretrained_ckpt/best_model_dim128.pth \
  --data_manifest manifests/qm7_train_100.txt \
  --target_property atomization_energy \
  --epochs 1 \
  --batch_size 8 \
  --device cpu \
  --save_model outputs/smoke_qm7_finetune_after_conformers/best_model.pth
```

SMILES multi-conformer smoke test:

```bash
python fine-tune/fine_tune_training.py \
  --pretrained_checkpoint_path pretrained_ckpt/best_model_dim128.pth \
  --data_root outputs/smoke_smiles.csv \
  --target_property toy_property \
  --num_conformers 3 \
  --val_split 0.4 \
  --epochs 1 \
  --batch_size 2 \
  --device cpu \
  --save_model outputs/smoke_smiles_finetune/best_model.pth
```

Verified in this update:

- the original QM7 `xyz` downstream path still trains and evaluates correctly
- CSV/SMILES fine-tuning with RDKit conformer generation completed successfully
- conformer-averaged evaluation completed successfully with `fine-tune/property_predict.py`

## Evaluation

### Atomic Charge Evaluation

```bash
python pretrain/charge_predict.py \
  --checkpoint_path pretrained_ckpt/best_model_dim128.pth \
  --test_data_root data/QM9/test_database.lmdb \
  --batch_size 8 \
  --device cpu \
  --charges_dir outputs/charge_eval/charges \
  --plot_path outputs/charge_eval/charge_scatter.png \
  --metrics_json outputs/charge_eval/charge_metrics.json
```

### Property Evaluation

QM7:

```bash
python fine-tune/property_predict.py \
  --checkpoint fine-tuned_ckpt/qm7/qm7_data300.pth \
  --test_data_root data/QM7/test_database \
  --device cpu \
  --batch_size 16 \
  --plot_dir outputs/qm7_eval/plots \
  --metrics_json outputs/qm7_eval/metrics.json
```

QM9 dipole:

```bash
python fine-tune/property_predict.py \
  --checkpoint fine-tuned_ckpt/dipole/dipole_data1000.pth \
  --test_data_root data/QM9/test_database \
  --device cpu \
  --batch_size 16 \
  --plot_dir outputs/dipole_eval/plots \
  --metrics_json outputs/dipole_eval/metrics.json
```

SMILES CSV with the conformer settings saved in the checkpoint:

```bash
python fine-tune/property_predict.py \
  --checkpoint outputs/my_smiles_model.pth \
  --test_data_root my_smiles_dataset.csv \
  --device cpu \
  --batch_size 16 \
  --plot_dir outputs/my_smiles_eval/plots \
  --metrics_json outputs/my_smiles_eval/metrics.json
```

## Verified Results in This Update

All values below were re-evaluated with the current code in the `chemprop` environment on **March 31, 2026**.

### Pretrained Checkpoint

| Checkpoint | Dataset | Metric |
|---|---|---|
| `best_model_dim128.pth` | QM9 test atomic charges (`lmdb`) | MAE `0.06765`, MSE `0.01675`, R² `0.6963` |

### QM7 Checkpoints

| Checkpoint | Target | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| `qm7_data100.pth` | `atomization_energy` | 215.3193 | 267.0371 | -0.3952 |
| `qm7_data300.pth` | `atomization_energy` | 102.1142 | 135.4177 | 0.6412 |
| `qm7_data500.pth` | `atomization_energy` | 93.1684 | 121.8443 | 0.7095 |
| `qm7_data800.pth` | `atomization_energy` | 86.8120 | 121.5446 | 0.7110 |
| `qm7_data1000.pth` | `atomization_energy` | 70.2080 | 90.5087 | 0.8397 |

### QM9 Dipole Checkpoints

| Checkpoint | Target | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| `dipole_data1000.pth` | `dipole` | 1.0547 | 1.7378 | 0.1927 |
| `dipole_data5000.pth` | `dipole` | 0.7664 | 1.4047 | 0.4726 |
| `dipole_data20000.pth` | `dipole` | 0.4015 | 0.9527 | 0.7574 |
| `dipole_data100000.pth` | `dipole` | 0.1999 | 0.4544 | 0.9448 |

### Smoke-Test Metrics Added in This Update

- LMDB pretraining smoke test on `data/QM9/test_database.lmdb`: validation loss `0.010624`, validation R² `0.8086`
- CSV/SMILES multi-conformer smoke test: evaluation MAE `1.817654`, RMSE `1.904044`, R² `-39.1038`
- QM7 xyz regression smoke test after the conformer update: evaluation MAE `1535.269531`, RMSE `1551.813616`, R² `-46.1156`

## Convenience Launcher

Common workflows are wrapped in `scripts/run_experiment.py`.

Examples:

```bash
python scripts/run_experiment.py prepare-splits
python scripts/run_experiment.py build-qm9-lmdb
python scripts/run_experiment.py pretrain-qm9 --epochs 100 --device cpu
python scripts/run_experiment.py finetune-qm7 --subset_size 300 --epochs 100 --device cpu
python scripts/run_experiment.py finetune-qm9-dipole --subset_size 1000 --epochs 100 --device cpu
```

## Notes

- `alignment_analysis/` additionally requires `seaborn`:

```bash
pip install -r requirements-optional.txt
```

- The long-running paper reproduction experiments were not retrained to completion during this maintenance pass, but the released checkpoints, evaluation commands, LMDB conversion path, deterministic subset generation, and all new smoke tests were validated end to end.
- If you want to benchmark your local environment first, run the evaluation commands above before launching a full retraining job.
