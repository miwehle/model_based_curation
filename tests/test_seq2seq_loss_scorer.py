from __future__ import annotations

import math

import pytest
import torch

from model_based_curation import Seq2SeqLossScorer


class _FakeSeq2Seq(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits
        self.forward_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        self.forward_calls.append((src.clone(), tgt.clone()))
        return self._logits.to(src.device)


def test_seq2seq_loss_scorer_matches_translator_teacher_forcing_loss():
    logits = torch.tensor([[
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
    ]], dtype=torch.float)
    model = _FakeSeq2Seq(logits)
    scorer = Seq2SeqLossScorer(model, device="cpu", tgt_pad_id=0)

    loss = scorer({"src_ids": [10, 11], "tgt_ids": [1, 1, 2, 3]})

    expected = float(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            torch.tensor([1, 2, 3]),
            ignore_index=0,
        ).item()
    )
    assert loss == pytest.approx(expected)
    assert len(model.forward_calls) == 1
    assert model.forward_calls[0][0].tolist() == [[10, 11]]
    assert model.forward_calls[0][1].tolist() == [[1, 1, 2, 3]]


def test_seq2seq_loss_scorer_ignores_tgt_padding_tokens():
    logits = torch.tensor([[
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
        [4.0, 0.0, 0.0],
    ]], dtype=torch.float)
    model = _FakeSeq2Seq(logits)
    scorer = Seq2SeqLossScorer(model, device="cpu", tgt_pad_id=0)

    loss = scorer({"src_ids": [10], "tgt_ids": [1, 1, 2, 0]})

    expected = float(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            torch.tensor([1, 2, 0]),
            ignore_index=0,
        ).item()
    )
    assert loss == pytest.approx(expected)
    assert math.isfinite(loss)
