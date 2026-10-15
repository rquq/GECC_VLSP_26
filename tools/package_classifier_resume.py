"""Build a private data/checkpoint upload for interrupted classifier ensembles."""

import argparse
from model.classifier_resume import package, sha256, MANIFEST
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output", required=True, help="New packet directory; ZIP is written beside it"
    )
    args = parser.parse_args()
    archive = package(args.source, args.dataset, args.output)
    print(f"Archive: {archive}")
    print(f"Manifest SHA256: {sha256(Path(args.output) / MANIFEST)}")


if __name__ == "__main__":
    main()
