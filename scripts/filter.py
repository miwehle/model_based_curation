from __future__ import annotations

from pathlib import Path

from _bootstrap import add_src_dirs, configure_logging

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure import read_run_config

    from model_based_curation import FilterConfig, filter

    configure_logging()
    if len(sys.argv) != 2:
        print("Usage: python scripts/filter.py <config-path>")
        return 1

    try:
        cfg = read_run_config(Path(sys.argv[1]))
        filter(FilterConfig(**cfg))
    except Exception as exc:
        print(f"Filter failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

