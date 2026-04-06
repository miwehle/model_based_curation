from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def collate_examples(
    batch: list[Mapping[str, Any]],
    *,
    id_field: str,
    src_field: str,
    tgt_field: str,
    src_pad_id: int,
    tgt_pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    max_src = max(len(list(example[src_field])) for example in batch)
    max_tgt = max(len(list(example[tgt_field])) for example in batch)
    bsz = len(batch)

    src_batch = torch.full((bsz, max_src), src_pad_id, dtype=torch.long)
    tgt_batch = torch.full((bsz, max_tgt), tgt_pad_id, dtype=torch.long)
    batch_ids: list[int] = []

    for i, example in enumerate(batch):
        ex_id = int(example[id_field])
        src_ids = [int(x) for x in example[src_field]]
        tgt_ids = [int(x) for x in example[tgt_field]]
        batch_ids.append(ex_id)
        src_batch[i, : len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        tgt_batch[i, : len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)
    return src_batch, tgt_batch, batch_ids
