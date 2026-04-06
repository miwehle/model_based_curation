from __future__ import annotations

from model_based_curation.collate import collate_examples


def test_collate_examples_pads_src_and_tgt_and_returns_batch_ids():
    src, tgt, batch_ids = collate_examples(
        [
            {"id": 100, "src_ids": [1, 2, 3], "tgt_ids": [9, 8]},
            {"id": 101, "src_ids": [4], "tgt_ids": [7, 6, 5]},
        ],
        id_field="id",
        src_field="src_ids",
        tgt_field="tgt_ids",
        src_pad_id=0,
        tgt_pad_id=0,
    )

    assert tuple(src.shape) == (2, 3)
    assert tuple(tgt.shape) == (2, 3)
    assert batch_ids == [100, 101]
    assert src.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert tgt.tolist() == [[9, 8, 0], [7, 6, 5]]
