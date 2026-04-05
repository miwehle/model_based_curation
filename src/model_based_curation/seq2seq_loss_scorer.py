from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class Seq2SeqLossScorer:
    """Per-example loss adapter matching the translator Seq2Seq training loss."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device,
        tgt_pad_id: int,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)

    def __call__(self, example: Mapping[str, Any]) -> float:
        src = self._tensor(example["src_ids"])
        tgt = self._tensor(example["tgt_ids"])
        self._model.eval()
        with torch.no_grad():
            logits = self._model(src, tgt)
            loss = self._criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
        return float(loss.item())

    def _tensor(self, token_ids: Any) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self._device)
