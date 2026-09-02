"""`python -m cacheverifier` / the `cacheverifier` console script.

A thin argparse dispatcher. The base install (httpx only) provides
`--version` and `--help`; the `healthcheck` subcommand additionally needs
the `healthcheck` extra and imports nothing heavy until it runs.
"""

from __future__ import annotations

import argparse
import sys

from cacheverifier import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cacheverifier",
        description="CacheVerifier -- hosted semantic-cache verification (https://www.cacheverifier.com).",
    )
    parser.add_argument("--version", action="version", version=f"cacheverifier {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Import lazily and defensively: a missing healthcheck extra must not
    # break `cacheverifier --help` or `--version`. cli.add_subparser itself
    # only touches argparse.
    from cacheverifier._healthcheck import cli as healthcheck_cli

    healthcheck_cli.add_subparser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
