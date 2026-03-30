from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from embedding2property import MolecularTransformer
from comenet4charge import ComENetAutoEncoder
from met_utils import build_pretrained_model


class FineTunedModel(nn.Module):
    def __init__(
        self,
        pretrained_checkpoint_path: str,
        device: torch.device,
        molecular_transformer_args: Optional[Dict] = None,
    ):
        super().__init__()
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
        if molecular_transformer_args is None:
            molecular_transformer_args = checkpoint["molecular_transformer_args"]

        state_dict = checkpoint.get("model_state_dict", {})
        autoencoder_only_checkpoint = checkpoint
        if any(key.startswith("autoencoder.") for key in state_dict):
            autoencoder_only_checkpoint = dict(checkpoint)
            autoencoder_only_checkpoint["model_state_dict"] = {
                key[len("autoencoder."):]: value
                for key, value in state_dict.items()
                if key.startswith("autoencoder.")
            }

        self.autoencoder, self.pretrained_config = build_pretrained_model(
            ComENetAutoEncoder,
            autoencoder_only_checkpoint,
            device,
            out_channels=1,
        )
        self.molecular_transformer = MolecularTransformer(**molecular_transformer_args).to(device)

        if any(key.startswith("autoencoder.") for key in state_dict):
            autoencoder_state = {
                key[len("autoencoder."):]: value
                for key, value in state_dict.items()
                if key.startswith("autoencoder.")
            }
            transformer_state = {
                key[len("molecular_transformer."):]: value
                for key, value in state_dict.items()
                if key.startswith("molecular_transformer.")
            }
            self.autoencoder.load_state_dict(autoencoder_state)
            self.molecular_transformer.load_state_dict(transformer_state)

    def forward(self, data):
        atomic_embeddings, _ = self.autoencoder._forward(data)
        embeddings = self.autoencoder.encoder(atomic_embeddings)
        padded_embeddings, mask = to_dense_batch(embeddings, batch=data.batch)
        return self.molecular_transformer(padded_embeddings, mask.float())
