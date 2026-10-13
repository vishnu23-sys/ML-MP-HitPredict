"""Command-line demo: predict Billboard Hot 100 success for held-out test songs.

Run `python scripts/run_all.py` (or `--only improvements`) first to create the models.

Examples:
    python -m hitpredict.predict --random 5
    python -m hitpredict.predict --search "lenny kravitz"
    python -m hitpredict.predict --index 12
    python -m hitpredict.predict --song ArtistScore=1 Danceability=0.8 Energy=0.7 Loudness=-5
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from .config import ALL_FEATURES, MODEL_DIR

DEFAULTS = {  # training-set medians (rounded), used to fill unspecified --song fields
    "ArtistScore": 0, "Danceability": 0.582, "Energy": 0.69, "Speechiness": 0.047,
    "Acousticness": 0.085, "Instrumentalness": 0.0, "Liveness": 0.13, "Valence": 0.52,
    "Loudness": -7.18, "Tempo": 119.2, "Key": 6, "Mode": 1,
}


def load_bundle():
    paths = {"best": MODEL_DIR / "best_model.joblib", "paper": MODEL_DIR / "paper_lr.joblib",
             "test": MODEL_DIR / "test_songs.csv"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing " + ", ".join(missing) +
                                ". Run: python scripts/run_all.py --only improvements")
    return (joblib.load(paths["best"]), joblib.load(paths["paper"]),
            pd.read_csv(paths["test"]))


def predict_frame(bundle, df: pd.DataFrame) -> np.ndarray:
    X = df[bundle["features"]].to_numpy(dtype=float)
    return bundle["model"].predict_proba(X)[:, 1]


def score_songs(best, paper, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_best"] = predict_frame(best, df)
    out["p_paper"] = predict_frame(paper, df)
    out["pred_best"] = (out.p_best >= 0.5).astype(int)
    thr = best.get("threshold_precision80")
    if thr is not None:
        out["pred_cautious"] = (out.p_best >= thr).astype(int)
    return out


def _fmt_row(r, thr):
    truth = {1: "HIT", 0: "non-hit"}.get(r.get("Label"), "?")
    line = (f"{r['Artist'][:28]:28s} | {r['Track'][:34]:34s} | artist score {int(r['ArtistScore'])}"
            f" | best model P(hit)={r['p_best']:.2f} | paper LR P(hit)={r['p_paper']:.2f}")
    if thr is not None:
        line += f" | cautious call: {'hit' if r['p_best'] >= thr else 'no'}"
    if "Label" in r and not pd.isna(r["Label"]):
        ok = "correct" if int(r["p_best"] >= 0.5) == int(r["Label"]) else "WRONG"
        line += f" | truth: {truth} ({ok})"
    return line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--random", type=int, metavar="K", help="score K random test songs")
    g.add_argument("--index", type=int, help="score one test song by row number")
    g.add_argument("--search", help="score test songs whose artist or title contains this text")
    g.add_argument("--song", nargs="+", metavar="FEATURE=VALUE",
                   help="score a custom song; unspecified features use typical values")
    g.add_argument("--summary", action="store_true", help="accuracy of both models on the test set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    best, paper, test = load_bundle()
    thr = best.get("threshold_precision80")
    print(f"Best model: {best['name']}  (grouped-CV accuracy {best['cv_acc']:.1%}, "
          f"test accuracy {best['test_acc']:.1%})")
    if thr is not None:
        print(f"Cautious threshold (80% precision on training CV): {thr:.2f}")
    print()

    if args.song:
        song = dict(DEFAULTS)
        for kv in args.song:
            k, v = kv.split("=", 1)
            if k not in DEFAULTS:
                sys.exit(f"Unknown feature {k}. Choose from: {', '.join(ALL_FEATURES)}")
            song[k] = float(v)
        df = pd.DataFrame([song]).assign(Artist="(custom)", Track="(custom)")
        print(_fmt_row(score_songs(best, paper, df).iloc[0], thr))
        return

    scored = score_songs(best, paper, test)
    if args.summary:
        for col, name in [("p_best", best["name"]), ("p_paper", paper["name"])]:
            acc = np.mean((scored[col] >= 0.5) == scored.Label)
            print(f"{name}: test accuracy {acc:.1%} on {len(scored)} songs by unseen artists")
        return
    if args.random:
        rows = scored.sample(args.random, random_state=args.seed)
    elif args.index is not None:
        rows = scored.iloc[[args.index]]
    else:
        q = args.search.lower()
        rows = scored[scored.Artist.str.contains(q, regex=False)
                      | scored.Track.str.contains(q, regex=False)]
        if rows.empty:
            sys.exit(f"No test song matches '{args.search}'")
    for _, r in rows.iterrows():
        print(_fmt_row(r, thr))


if __name__ == "__main__":
    main()
