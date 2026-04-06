from __future__ import annotations

from pathlib import Path

from model_based_curation import SplitConfig


def test_split_config_derives_conventional_paths_from_dataset_and_checkpoint():
    cfg = SplitConfig(
        dataset="iwslt2017_iwslt2017-de-en_train",
        checkpoint="2eu_10tt_1eu_5tt_5nc_1eu_5tt",
        upper_bounds=(0.5, 1.5),
        batch_size=64,
        sort_by_loss_desc=True,
    )

    assert cfg.dataset_drive_path == Path(
        "/content/drive/MyDrive/nmt_lab/artifacts/datasets/iwslt2017_iwslt2017-de-en_train"
    )
    assert cfg.dataset_local_path == Path(
        "/content/nmt_lab/artifacts/iwslt2017_iwslt2017-de-en_train"
    )
    assert cfg.output_path == Path(
        "/content/nmt_lab/artifacts/iwslt2017_iwslt2017-de-en_train/curation/loss_buckets"
    )
    assert cfg.drive_output_path == Path(
        "/content/drive/MyDrive/nmt_lab/artifacts/datasets/iwslt2017_iwslt2017-de-en_train/curation/loss_buckets"
    )
    assert cfg.checkpoint_file == Path(
        "/content/drive/MyDrive/nmt_lab/artifacts/training_runs/2eu_10tt_1eu_5tt_5nc_1eu_5tt/checkpoint.pt"
    )
    assert cfg.sort_by_loss_desc is True


def test_split_config_uses_german_csv_defaults():
    cfg = SplitConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5))

    assert cfg.csv_delimiter == ";"
    assert cfg.loss_decimal_separator == ","


def test_split_config_validates_csv_format_options():
    try:
        SplitConfig(
            dataset="dataset",
            checkpoint="run",
            upper_bounds=(0.5, 1.5),
            csv_delimiter="|",
        )
    except ValueError as exc:
        assert str(exc) == "csv_delimiter must be ',' or ';'."
    else:
        raise AssertionError("Expected ValueError for unsupported csv_delimiter.")

    try:
        SplitConfig(
            dataset="dataset",
            checkpoint="run",
            upper_bounds=(0.5, 1.5),
            loss_decimal_separator=":",
        )
    except ValueError as exc:
        assert str(exc) == "loss_decimal_separator must be '.' or ','."
    else:
        raise AssertionError(
            "Expected ValueError for unsupported loss_decimal_separator."
        )

    try:
        SplitConfig(
            dataset="dataset",
            checkpoint="run",
            upper_bounds=(0.5, 1.5),
            decode_from_loss=-0.1,
        )
    except ValueError as exc:
        assert str(exc) == "decode_from_loss must be non-negative."
    else:
        raise AssertionError("Expected ValueError for negative decode_from_loss.")
