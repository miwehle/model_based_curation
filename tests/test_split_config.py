from __future__ import annotations

from pathlib import Path

from model_based_curation import FilterConfig, SplitConfig


def test_configs_use_local_datasets_directory():
    assert SplitConfig(
        dataset="dataset",
        checkpoint="run",
        upper_bounds=(0.5, 1.5),
    ).dataset_local_path == Path("/content/nmt_lab/artifacts/datasets/dataset")
    assert FilterConfig(dataset="dataset").dataset_local_path == Path(
        "/content/nmt_lab/artifacts/datasets/dataset"
    )


def test_split_config_validates_csv_format_options():
    try:
        SplitConfig(
            dataset="dataset",
            checkpoint="run",
            upper_bounds=(0.5, 1.5),
            log_every_batches=0,
        )
    except ValueError as exc:
        assert str(exc) == "log_every_batches must be positive."
    else:
        raise AssertionError("Expected ValueError for non-positive log_every_batches.")

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

    try:
        SplitConfig(
            dataset="dataset",
            checkpoint="run",
            upper_bounds=(0.5, 1.5),
            decode_at_least=-1,
        )
    except ValueError as exc:
        assert str(exc) == "decode_at_least must be non-negative."
    else:
        raise AssertionError("Expected ValueError for negative decode_at_least.")
