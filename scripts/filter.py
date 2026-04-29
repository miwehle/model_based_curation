from __future__ import annotations

from _bootstrap import add_src_dirs, configure_logging

add_src_dirs(__file__)


def main() -> int:
    from lab_infrastructure import run_config_cli

    from model_based_curation import FilterConfig, filter

    configure_logging()
    run_config_cli(filter, FilterConfig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
