"""Download, load and split the HitPredict dataset.

The data is the authors' own: about 4,000 songs from 1990-2018, half Billboard Hot 100 hits and
half random Million Song Dataset tracks, with Spotify audio features. It is distributed inside the
code archive linked from the paper. The Spotify audio-features endpoint the authors used was closed
to new applications in November 2024, so this archive is the only way to get the original features.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config


# --------------------------------------------------------------------------- download

def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_present() -> bool:
    return config.DATED_CSV.exists() and config.UNDATED_CSV.exists()


def download(force: bool = False, timeout: int = 60) -> None:
    """Fetch the authors' archive and extract only the two CSV files into data/raw."""
    import requests

    if data_present() and not force:
        print(f"Data already present in {config.RAW_DIR}")
        verify()
        return
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.DATA_ZIP_URL}")
    resp = requests.get(config.DATA_ZIP_URL, timeout=timeout)
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):
        raise RuntimeError(
            "The download did not return a zip archive. Open https://tinyurl.com/yb6cvtek in a "
            "browser, download 'Burton Suta Georgieva Code.zip' and copy its two CSV files into "
            f"{config.RAW_DIR}."
        )
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in config.CSV_SHA256:
            member = next(m for m in zf.namelist() if m.endswith(name))
            (config.RAW_DIR / name).write_bytes(zf.read(member))
    verify()


def verify() -> None:
    for name, expected in config.CSV_SHA256.items():
        got = _sha256(config.RAW_DIR / name)
        if got != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {got}")
    print("Checksums OK")


# --------------------------------------------------------------------------- loading

_SPLIT_ARTIST = re.compile(r"\s+(?:feat\.?|featuring|ft\.?|with|x|&|and|vs\.?)\s+|,|/")


def primary_artist(name: str) -> str:
    """Lead artist used as the grouping key, so 'drake feat. rihanna' groups with 'drake'."""
    return _SPLIT_ARTIST.split(str(name).strip().lower())[0].strip()


def load_raw_dated() -> pd.DataFrame:
    """The authors' dated file exactly as published, in file order (used for replication)."""
    return pd.read_csv(config.DATED_CSV)


def load_raw_undated() -> pd.DataFrame:
    return pd.read_csv(config.UNDATED_CSV)


def load_master() -> pd.DataFrame:
    """All labelled songs, de-duplicated, with release year/month where the authors had it.

    The undated file is a strict superset of the dated one (every dated song is in it with the
    same features and label), so it is the master table. Year and Month are NaN for the 813 songs
    whose Spotify release date had no month.
    """
    und = load_raw_undated().drop_duplicates().reset_index(drop=True)
    dated = load_raw_dated().drop_duplicates()
    keyed = dated[["Artist", "Track", "Year", "Month"]].drop_duplicates(["Artist", "Track"])
    df = und.merge(keyed, on=["Artist", "Track"], how="left")
    df["Group"] = df["Artist"].map(primary_artist)
    df["Dated"] = df["Year"].notna()
    df.insert(0, "SongId", np.arange(len(df)))
    return df


# --------------------------------------------------------------------------- splits

def group_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = config.SEED,
                group_col: str = "Group"):
    """Train/test split where every artist lands on exactly one side, stratified on the label."""
    n_splits = int(round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[config.TARGET], df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def random_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = config.SEED):
    """Row-level stratified split (what the paper effectively did). Artists leak across sides."""
    tr, te = train_test_split(df, test_size=test_size, random_state=seed,
                              stratify=df[config.TARGET])
    return tr.copy(), te.copy()


def paper_split(df: pd.DataFrame, n_train: int = 2416):
    """The authors' split: first 2,416 rows of the file for training, the remaining 805 for validation."""
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def main_split():
    """The single fixed train/test split used for every headline number in this project."""
    df = load_master()
    return group_split(df, test_size=0.25, seed=config.SEED)


def summary() -> dict:
    """Dataset facts recorded in results.json (used by the README and the write-up)."""
    und, dated, df = load_raw_undated(), load_raw_dated(), load_master()
    tr, te = main_split()
    instr = df[df.Instrumentalness > 0.5]
    return {
        "raw_rows_undated_file": len(und), "raw_rows_dated_file": len(dated),
        "exact_duplicates_undated_file": int(und.duplicated().sum()),
        "songs": len(df), "hits": int(df.Label.sum()), "hit_rate": float(df.Label.mean()),
        "songs_with_month": int(df.Dated.sum()), "songs_without_month": int((~df.Dated).sum()),
        "lead_artists": int(df.Group.nunique()), "year_min": int(df.Year.min()),
        "year_max": int(df.Year.max()), "train_songs": len(tr), "test_songs": len(te),
        "train_lead_artists": int(tr.Group.nunique()), "test_lead_artists": int(te.Group.nunique()),
        "songs_instrumentalness_gt_0.5": len(instr),
        "hit_rate_instrumentalness_gt_0.5": float(instr.Label.mean()),
    }


def xy(df: pd.DataFrame, features):
    return df[list(features)].to_numpy(dtype=float), df[config.TARGET].to_numpy(dtype=int)
