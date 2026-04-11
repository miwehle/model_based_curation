from __future__ import annotations

import pytest
import torch

from model_based_curation.split.batch_seq2seq_loss_scorer import BatchSeq2SeqLossScorer


class _FakeSeq2Seq(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits
        self.forward_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        self.forward_calls.append((src.clone(), tgt.clone()))
        return self._logits.to(src.device)


def test_batch_seq2seq_loss_scorer_returns_mean_token_loss_per_example():
    logits = torch.tensor(
        [
            [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 4.0]],
            [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float,
    )
    model = _FakeSeq2Seq(logits)
    scorer = BatchSeq2SeqLossScorer(model, device="cpu", src_pad_id=0, tgt_pad_id=0)

    losses = scorer.score_batch(
        [
            {"id": 1, "src_ids": [10, 11], "tgt_ids": [1, 1, 2, 3]},
            {"id": 2, "src_ids": [12], "tgt_ids": [1, 1, 2]},
        ]
    )

    expected_1 = float(
        torch.nn.functional.cross_entropy(
            logits[0:1].reshape(-1, logits.size(-1)), torch.tensor([1, 2, 3]), ignore_index=0
        ).item()
    )
    expected_2 = float(
        torch.nn.functional.cross_entropy(
            logits[1:2].reshape(-1, logits.size(-1)), torch.tensor([1, 2, 0]), ignore_index=0
        ).item()
    )
    assert losses == pytest.approx([expected_1, expected_2])
    assert len(model.forward_calls) == 1
    assert model.forward_calls[0][0].tolist() == [[10, 11], [12, 0]]
    assert model.forward_calls[0][1].tolist() == [[1, 1, 2, 3], [1, 1, 2, 0]]
