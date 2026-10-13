"""Download the HitPredict dataset (the authors' published CSVs) into data/raw.

Usage:
    python scripts/download_data.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitpredict import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if files exist")
    args = ap.parse_args()
    data.download(force=args.force)
    df = data.load_master()
    print(f"Loaded {len(df)} unique songs ({int(df.Label.sum())} hits), "
          f"{int(df.Dated.sum())} with release month, {df.Group.nunique()} lead artists")


if __name__ == "__main__":
    main()
