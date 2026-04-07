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
    rows = [
        (
            int(example[id_field]),
            [int(x) for x in example[src_field]],
            [int(x) for x in example[tgt_field]],
        )
        for example in batch
    ]
    max_src = max(len(src_ids) for _, src_ids, _ in rows)
    max_tgt = max(len(tgt_ids) for _, _, tgt_ids in rows)
    bsz = len(batch)

    src_batch = torch.full((bsz, max_src), src_pad_id, dtype=torch.long)
    tgt_batch = torch.full((bsz, max_tgt), tgt_pad_id, dtype=torch.long)
    batch_ids: list[int] = []

    for i, (ex_id, src_ids, tgt_ids) in enumerate(rows):
        batch_ids.append(ex_id)
        src_batch[i, : len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        tgt_batch[i, : len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)
    return src_batch, tgt_batch, batch_ids
