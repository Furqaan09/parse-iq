"""Admin commands, run from api/:  uv run python -m app.cli reindex"""

import argparse
import logging

from sqlmodel import Session

from app.db import engine
from app.ingestion.indexing import reindex_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reindex", help="Re-chunk every document and rebuild the vector index")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "reindex":
        with Session(engine) as session:
            summary = reindex_all(session)
        print(summary)
        # A non-zero exit code lets scripts and CI notice a partial failure
        raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
