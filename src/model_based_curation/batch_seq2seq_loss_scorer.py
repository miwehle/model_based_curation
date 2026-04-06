from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from model_based_curation.collate import collate_examples


class BatchSeq2SeqLossScorer:
    """Batch scorer for per-example Seq2Seq loss values."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device,
        src_pad_id: int,
        tgt_pad_id: int,
        id_field: str = "id",
        src_field: str = "src_ids",
        tgt_field: str = "tgt_ids",
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._src_pad_id = src_pad_id
        self._tgt_pad_id = tgt_pad_id
        self._id_field = id_field
        self._src_field = src_field
        self._tgt_field = tgt_field
        self._criterion = nn.CrossEntropyLoss(
            ignore_index=tgt_pad_id, reduction="none"
        )

    def score_batch(self, examples: list[Mapping[str, Any]]) -> list[float]:
        src, tgt, _ = collate_examples(
            examples,
            id_field=self._id_field,
            src_field=self._src_field,
            tgt_field=self._tgt_field,
            src_pad_id=self._src_pad_id,
            tgt_pad_id=self._tgt_pad_id,
        )
        src = src.to(self._device)
        tgt = tgt.to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(src, tgt)
            per_token_loss = self._criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            ).reshape(tgt.size(0), -1)
        mask = (tgt[:, 1:] != self._tgt_pad_id).to(per_token_loss.dtype)
        loss_sums = (per_token_loss * mask).sum(dim=1)
        token_counts = mask.sum(dim=1)
        if torch.any(token_counts == 0):
            raise ValueError("Each example must contribute at least one non-pad target token.")
        return [float(x) for x in (loss_sums / token_counts).tolist()]
