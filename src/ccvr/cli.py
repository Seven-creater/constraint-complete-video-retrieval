from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dataset import audit_dataset, fetch_metadata
from .decision import audit_literature, finalize_direction
from .features import stream_extract_features
from .io import read_json
from .scoring import score_backbone


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccvr",
        description="Preregistered constraint-complete video retrieval direction gate.",
    )
    parser.add_argument(
        "--config",
        default="config/preregistered.json",
        help="Frozen preregistration JSON.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch-metadata")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--revision")

    audit = commands.add_parser("audit-dataset")
    audit.add_argument("--raw", required=True)
    audit.add_argument("--output", required=True)

    literature = commands.add_parser("audit-literature")
    literature.add_argument("--matrix", default="docs/literature_collision_matrix.csv")
    literature.add_argument("--output", required=True)

    features = commands.add_parser("fetch-features")
    features.add_argument("--backbone", choices=("OpenCLIP", "EVA-CLIP"), required=True)
    features.add_argument("--output", required=True)
    features.add_argument("--gate", required=True)

    score = commands.add_parser("score-backbone")
    score.add_argument("--backbone", choices=("OpenCLIP", "EVA-CLIP"), required=True)
    score.add_argument("--raw", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--features", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda:0")

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--run", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = read_json(Path(args.config))
    if args.command == "fetch-metadata":
        revision = args.revision or str(config["dataset"]["revision"])
        result = fetch_metadata(Path(args.output), revision=revision)
    elif args.command == "audit-dataset":
        result = audit_dataset(Path(args.raw), Path(args.output), config)
    elif args.command == "audit-literature":
        result = audit_literature(Path(args.matrix), Path(args.output))
    elif args.command == "fetch-features":
        result = stream_extract_features(
            args.backbone,
            Path(args.output),
            Path(args.gate),
            float(config["budget"]["maximum_new_storage_gb"]),
        )
    elif args.command == "score-backbone":
        result = score_backbone(
            args.backbone,
            Path(args.raw),
            Path(args.manifest),
            Path(args.features),
            Path(args.output),
            config,
            args.device,
        )
    elif args.command == "finalize":
        result = finalize_direction(Path(args.run))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
