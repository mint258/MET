from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


QM9_PROPERTY_NAMES = [
    "rot_A",
    "rot_B",
    "rot_C",
    "dipole",
    "polarizability",
    "HOMO_energy",
    "LUMO_energy",
    "gap",
    "R2",
    "zpve",
    "U0",
    "U298",
    "H298",
    "G298",
    "Cv",
]

QM7_PROPERTY_NAMES = ["atomization_energy"]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Dict) -> None:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_manifest_paths(manifest_path: str | Path) -> List[Path]:
    manifest_path = Path(manifest_path).resolve()
    files: List[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            parent_candidate = (manifest_path.parent / candidate).resolve()
            repo_candidate = (manifest_path.parent.parent / candidate).resolve()
            candidate = parent_candidate if parent_candidate.exists() else repo_candidate
        files.append(candidate)
    return files


def collect_files(
    root: Optional[str | Path] = None,
    manifest_path: Optional[str | Path] = None,
    suffixes: Iterable[str] = (".xyz",),
) -> List[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    if manifest_path:
        files = [p for p in _read_manifest_paths(manifest_path) if p.suffix.lower() in suffixes]
    else:
        if root is None:
            raise ValueError("Either root or manifest_path must be provided.")
        root = Path(root).resolve()
        if root.is_file():
            files = [root] if root.suffix.lower() in suffixes else []
        else:
            files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    if not files:
        location = manifest_path if manifest_path else root
        raise FileNotFoundError(f"No files with suffixes {suffixes} were found in {location}.")
    return files


def infer_pretrain_checkpoint_config(checkpoint: Dict) -> Dict[str, int | float]:
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    hidden_channels = state_dict["emb_z.emb.weight"].shape[1]
    atom_embedding_dim = state_dict["encoder.0.weight"].shape[0]
    middle_channels = state_dict["interaction_blocks.0.lin_feature1.lin1.weight"].shape[0]

    interaction_indices = [
        int(match.group(1))
        for key in state_dict
        for match in [re.match(r"interaction_blocks\.(\d+)\.", key)]
        if match
    ]
    num_layers = max(interaction_indices) + 1 if interaction_indices else 4

    output_indices = [
        int(match.group(1))
        for key in state_dict
        for match in [re.match(r"output_layers_z\.(\d+)\.", key)]
        if match
    ]
    out_channels = max(output_indices) + 1 if output_indices else 1

    dense_indices = [
        int(match.group(1))
        for key in state_dict
        for match in [re.match(r"lins\.(\d+)\.", key)]
        if match
    ]
    num_output_layers = max(dense_indices) + 1 if dense_indices else 0

    transformer_layer_indices = [
        int(match.group(1))
        for key in state_dict
        for match in [re.match(r"transformer_decoders_z\.0\.layers\.(\d+)\.", key)]
        if match
    ]
    transformer_layers = max(transformer_layer_indices) + 1 if transformer_layer_indices else 1

    feature1_in = state_dict["interaction_blocks.0.lin_feature1.lin1.weight"].shape[1]
    feature2_in = state_dict["interaction_blocks.0.lin_feature2.lin1.weight"].shape[1]
    num_spherical = max(1, feature1_in // max(1, feature2_in))
    num_radial = max(1, feature2_in // max(1, num_spherical))

    return {
        "cutoff": checkpoint.get("cutoff", 8.0),
        "num_layers": checkpoint.get("num_layers", num_layers),
        "hidden_channels": checkpoint.get("hidden_channels", hidden_channels),
        "middle_channels": checkpoint.get("middle_channels", middle_channels),
        "atom_embedding_dim": checkpoint.get("atom_embedding_dim", atom_embedding_dim),
        "out_channels": checkpoint.get("out_channels", out_channels),
        "num_radial": checkpoint.get("num_radial", num_radial),
        "num_spherical": checkpoint.get("num_spherical", num_spherical),
        "num_output_layers": checkpoint.get("num_output_layers", max(1, num_output_layers)),
        "transformer_layers": checkpoint.get("transformer_layers", transformer_layers),
        "nhead_z": checkpoint.get("nhead_z", 1),
    }


def build_pretrained_model(model_class, checkpoint: Dict, device: torch.device, **overrides):
    config = infer_pretrain_checkpoint_config(checkpoint)
    config.update(overrides)
    model = model_class(device=str(device), **config)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model.to(device), config
