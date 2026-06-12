# -*- coding: utf-8 -*-
"""
Per-modality reconstruction pretraining for PARK-IN-SON encoders.

WHY pretrain separately before end-to-end fine-tuning?

  * With small medical cohorts (~500 PPMI patients), training the full
    encoder+fusion+GNN end-to-end from scratch risks the downstream task
    signal (UPDRS regression) being too sparse to learn good per-modality
    representations.

  * Reconstruction pretraining is self-supervised: every patient can be used
    regardless of label completeness.  The encoder learns the *structure* of
    each modality before being asked to predict clinical outcomes.

  * After pretraining, the encoders are warm-started and only need fine-grained
    task-specific adjustments during end-to-end training.

Training strategy per modality
-------------------------------
  Clinical / MRI / PET  : MSE reconstruction loss (continuous features).
  Genetic               : Cross-entropy per gene (3-class: WT / mutation /
                          unknown) — uses ``GeneticEncoder.decode_logits()``.

Usage
-----
    from src.models import MultimodalEncoder
    from src.models.pretrain import pretrain_all

    model = MultimodalEncoder.from_config(config)
    histories = pretrain_all(model, data_tensors, config, device="cuda")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.multimodal_encoder import MultimodalEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-modality pretraining
# ---------------------------------------------------------------------------

def pretrain_modality(
    encoder: nn.Module,
    data: torch.Tensor,
    modality_name: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 10,
    device: str = "cpu",
    save_path: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """Pretrain one modality encoder using reconstruction loss.

    Args:
        encoder:       The modality encoder (``BaseEncoder`` subclass).
        data:          ``(N, ...)`` tensor of training data for this modality.
        modality_name: One of ``"clinical"``, ``"mri"``, ``"pet"``,
                       ``"genetic"``.  Determines loss function.
        epochs:        Maximum training epochs.
        lr:            AdamW learning rate.
        batch_size:    Mini-batch size.
        patience:      Early stopping patience (epochs without improvement).
        device:        ``"cpu"`` or ``"cuda"`` or ``"mps"``.
        save_path:     If provided, best weights are checkpointed here.

    Returns:
        ``{"train_loss": [float, ...]}``.
    """
    encoder = encoder.to(device)
    data = data.to(device)

    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    optimizer = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss function depends on modality type
    if modality_name == "genetic":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"train_loss": []}
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in loader:
            optimizer.zero_grad(set_to_none=True)

            if modality_name == "genetic":
                # Genetic: encode → decode → per-gene 3-class logits
                z = encoder.encode(batch_x)
                logits = encoder.decode_logits(z)          # (B, n_genes, 3)
                B, n_genes, n_classes = logits.shape
                target = batch_x.long().clamp(0, 2)        # (B, n_genes)
                loss = criterion(
                    logits.view(-1, n_classes), target.view(-1)
                )
            else:
                # All other modalities: AE reconstruction
                z, x_hat = encoder.forward_with_reconstruction(batch_x)
                loss = criterion(x_hat.view_as(batch_x), batch_x)

            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"[pretrain:{modality_name}] epoch {epoch+1:>3}/{epochs} "
                f"— loss: {avg_loss:.5f}  lr: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Early stopping + checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(encoder.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"[pretrain:{modality_name}] early stop at epoch {epoch+1} "
                    f"(best loss: {best_loss:.5f})"
                )
                break

    # Restore best checkpoint
    if save_path is not None and save_path.exists():
        encoder.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        logger.info(f"[pretrain:{modality_name}] best weights restored from {save_path}")

    return history


# ---------------------------------------------------------------------------
# Multi-modality pretraining coordinator
# ---------------------------------------------------------------------------

def pretrain_all(
    model: MultimodalEncoder,
    data_tensors: Dict[str, torch.Tensor],
    config: dict,
    save_dir: Path = Path("outputs/models/encoders"),
    device: str = "cpu",
) -> Dict[str, Dict[str, List[float]]]:
    """Pretrain all modality encoders sequentially.

    Each encoder is pretrained independently (no interaction with the fusion
    layer during pretraining).  After this function returns, all encoder
    weights in ``model`` are updated in-place with the best pretraining
    checkpoint.

    Args:
        model:        The ``MultimodalEncoder`` whose encoders will be
                      pretrained in-place.
        data_tensors: ``dict[modality_name -> (N, ...) tensor]``.
                      Only modalities present in this dict are pretrained.
        config:       Full project config dict (reads ``multimodal_encoder.
                      training`` section).
        save_dir:     Directory for per-encoder weight checkpoints.
        device:       Torch device string.

    Returns:
        ``dict[modality_name -> {"train_loss": [...]}]``.
    """
    mc_train = config.get("multimodal_encoder", {}).get("training", {})
    pretrain_epochs: int = mc_train.get("pretrain_epochs", 50)
    lr: float = mc_train.get("lr", 1e-3)
    batch_size: int = mc_train.get("batch_size", 32)
    patience: int = mc_train.get("patience", 10)

    histories: Dict[str, Dict[str, List[float]]] = {}

    for name, encoder in model.encoders.items():
        if name not in data_tensors:
            logger.warning(
                f"pretrain_all: no tensor for '{name}' — skipping encoder."
            )
            continue

        logger.info(
            f"=== Pretraining {name} encoder "
            f"({sum(p.numel() for p in encoder.parameters()):,} params) ==="
        )

        hist = pretrain_modality(
            encoder=encoder,
            data=data_tensors[name],
            modality_name=name,
            epochs=pretrain_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
            save_path=save_dir / f"{name}_encoder_pretrain.pt",
        )
        histories[name] = hist

    logger.info("=== All encoder pretraining complete ===")
    return histories
