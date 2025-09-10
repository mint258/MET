MET: Molecular Equivariant Transformer — Pretrain on Atomic Charges, Fine-Tune for Properties
============================================================================================

MET is a symmetry-aware pretraining framework for molecular representation learning.
It couples an Equivariant Graph Neural Network (EGNN) with Transformer layers and is
pretrained to predict atomic partial charges from 3D geometries. The pretrained encoder
can be fine-tuned to downstream molecular properties (e.g., QM7/QM9), and is especially
effective in low-data regimes.

Paper
------
Molecular Equivariant Transformer (MET) — ChemRxiv, 2025
Link: https://chemrxiv.org/engage/chemrxiv/article-details/689e8887a94eede154d606f4


Repository Layout
-----------------
.
├── LICENSE
├── MET.yml
├── data/
│   ├── QM9/
│   └── QM7/
├── pretrained_ckpt/                  # Provided pretrained weights (e.g., best_model_dim128.pth)
├── fine-tuned_ckpt/                  # Fine-tuned weights will be saved here
├── pretrain/                         # Pretraining: charges from 3D (EGNN backbone)
│   ├── comenet4charge.py             # Pretraining model definition
│   ├── features.py                   # Featurization utilities
│   ├── dataset_without_charge.py     # Dataloader for QM9 (charges as labels)
│   ├── training_charge_model.py      # Training script
│   ├── charge_predict.py             # Inference on new molecules; scatter of pred vs. ref
│   └── (artifacts: best_comenet_model.pth, embeddings/, charges/, ...)
├── fine-tune/                        # Fine-tuning: pool embeddings -> target property
│   ├── FineTunedModel.py
│   ├── embedding2property.py         # Heads / readouts for properties
│   ├── dataset_finetune.py           # Flexible I/O: XYZ or CSV (can RDKit-generate coords)
│   ├── fine_tune_training.py         # Fine-tuning script (requires pretrained checkpoint)
│   └── property_predict.py           # Inference & scatter plot for properties
├── alignment_analysis/               # Embedding visualization / clustering
│   ├── embedding_plot.py             # For custom XYZ sets
│   └── embedding_plot_qm9.py         # For QM9 splits
└── run_quickstart.sh                 # One-command demo (inference + visualization)


Requirements
------------
- Python 3.9+
- PyTorch (CUDA optional)
- RDKit (optional; needed only when generating 3D from SMILES/CSV during fine-tuning)
- Standard scientific Python stack (numpy, scipy, matplotlib, etc.)


Quick Start (No Training Required)
----------------------------------
1) Ensure minimal data paths exist:
   - data/QM9/test_database/
   - data/QM7/test_database/

2) Ensure a pretrained weight exists (already included):
   - pretrained_ckpt/best_model_dim128.pth

3) From the repository root, run:
   bash run_quickstart.sh

What the script does:
- Detects CUDA availability automatically.
- Runs charge prediction on QM9 test set using the pretrained model (pretrain/charge_predict.py).
- If a fine-tuned checkpoint exists under fine-tuned_ckpt/, also runs property prediction on QM7
  (fine-tune/property_predict.py).
- Produces example plots according to the scripts’ defaults or provided flags.


Pretraining (Charges from 3D)
------------------------------
All pretraining scripts live in pretrain/. Typical usage:

# Train (EGNN + Transformer, charges as labels):
cd pretrain
python training_charge_model.py \
  --data_root ../data/QM9/train_valid_database/ \
  --device cuda

# Inference & scatter (pred vs. ref) on QM9 test:
python charge_predict.py \
  --checkpoint_path best_comenet_model.pth \
  --test_data_root ../data/QM9/test_database/

Model code:
- comenet4charge.py (architecture)
- features.py (featurization)
Data loader:
- dataset_without_charge.py (QM9)


Fine-Tuning (Downstream Properties)
-----------------------------------
Scripts live in fine-tune/. You must provide a pretrained checkpoint.

Example fine-tuning on QM9:
cd fine-tune
python fine_tune_training.py \
  --pretrained_checkpoint_path ../pretrained_ckpt/best_model_dim128.pth \
  --data_root ../data/QM9/test_database/ \
  --target_property P7 P8 P9 \
  --dropout 0 --learning_rate 1e-4 \
  --freeze_up_to_layer 4 --num_layers 0 \
  --seed 114 --batch_size 32

Example fine-tuning on QM7 (small split):
cd fine-tune
python fine_tune_training.py \
  --pretrained_checkpoint_path ../pretrained_ckpt/best_model_dim128.pth \
  --data_root ../data/QM7/train_database_300/ \
  --target_property P2 \
  --dropout 0 --learning_rate 1e-4 \
  --freeze_up_to_layer 4 --num_layers 0 \
  --seed 114 --batch_size 32

Inference after fine-tuning (scatter of predicted vs. true):
cd fine-tune
python property_predict.py \
  --checkpoint ../fine-tuned_ckpt/qm7/qm7_data500.pth \
  --test_data_root ../data/QM7/test_database/ \
  --target_property P2 \
  --plot_path qm7.png

Notes:
- dataset_finetune.py supports (1) XYZ folders and (2) CSV without coordinates.
  When coordinates are missing, RDKit is used to generate 3D structures.


Embedding Visualization (Clustering / t-SNE)
--------------------------------------------
cd alignment_analysis

# For a custom XYZ folder:
python embedding_plot.py \
  --xyz_dir new_xyz_structures/ \
  --model_path ../pretrained_ckpt/best_model_dim128.pth \
  --device cuda --perplexity 12 --batch_size 50

# For QM9 test split + optional MW/dipole overlays:
python embedding_plot_qm9.py \
  --xyz_dir ../data/QM9/test_database/ \
  --model_path ../pretrained_ckpt/best_model_dim128.pth \
  --device cuda --perplexity 12 --batch_size 50 \
  --mw_plot --dipole_plot --binary_group_plots


Citation
--------
If you find this repository useful, please cite:
Molecular Equivariant Transformer (MET) — ChemRxiv, 2025.
https://chemrxiv.org/engage/chemrxiv/article-details/689e8887a94eede154d606f4


License
-------
This project is released under the license found in the LICENSE file.
