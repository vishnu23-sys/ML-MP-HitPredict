"""HitPredict demo. Run with:  streamlit run app/streamlit_app.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitpredict.config import AUDIO_FEATURES, FIG_DIR, RESULTS_JSON  # noqa: E402
from hitpredict.predict import DEFAULTS, load_bundle, score_songs  # noqa: E402

st.set_page_config(page_title="HitPredict", layout="wide")


@st.cache_resource
def bundle():
    best, paper, test = load_bundle()
    return best, paper, score_songs(best, paper, test)


try:
    best, paper, scored = bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

thr = best.get("threshold_precision80")
st.title("HitPredict: will this song reach the Billboard Hot 100?")
st.caption(
    "Spotify audio features plus an artist score (1 if the artist already had a Hot 100 hit). "
    f"All songs below are from the held-out test set: {len(scored)} songs by artists the models "
    "never saw during training.")

c1, c2, c3 = st.columns(3)
acc_best = np.mean(scored.pred_best == scored.Label)
acc_paper = np.mean((scored.p_paper >= 0.5) == scored.Label)
acc_rule = np.mean(scored.ArtistScore == scored.Label)
c1.metric("Best model, test accuracy", f"{acc_best:.1%}", help=best["name"])
c2.metric("Paper's LR (reproduced), test accuracy", f"{acc_paper:.1%}")
c3.metric("Artist score alone, test accuracy", f"{acc_rule:.1%}")

tab_song, tab_browse, tab_custom, tab_results = st.tabs(
    ["One test song", "Browse the test set", "Try your own song", "Experiment results"])

with tab_song:
    labels = (scored.Artist.str.title() + " - " + scored.Track.str.title()).tolist()
    idx = st.selectbox("Pick a test song (type to search)", range(len(scored)),
                       format_func=lambda i: labels[i])
    r = scored.iloc[idx]
    truth = "Hit" if r.Label == 1 else "Not a hit"
    a, b, c, d = st.columns(4)
    a.metric("Actual outcome", truth)
    b.metric("Best model: P(hit)", f"{r.p_best:.0%}",
             "correct" if r.pred_best == r.Label else "wrong",
             delta_color="normal" if r.pred_best == r.Label else "inverse")
    c.metric("Paper LR: P(hit)", f"{r.p_paper:.0%}")
    if thr is not None:
        d.metric(f"Cautious call (threshold {thr:.2f})",
                 "Hit" if r.p_best >= thr else "No call")
    st.write(f"Artist score: **{int(r.ArtistScore)}**"
             + (f", release {int(r.Year)}-{int(r.Month):02d}" if not pd.isna(r.Year) else ""))
    hits, non = scored[scored.Label == 1], scored[scored.Label == 0]
    prof = pd.DataFrame({
        "This song": [r[f] for f in AUDIO_FEATURES],
        "Median hit (test set)": [hits[f].median() for f in AUDIO_FEATURES],
        "Median non-hit (test set)": [non[f].median() for f in AUDIO_FEATURES],
    }, index=AUDIO_FEATURES)
    st.dataframe(prof.style.format("{:.3f}"), width="stretch")

with tab_browse:
    only_wrong = st.checkbox("Show only songs the best model gets wrong")
    view = scored[scored.pred_best != scored.Label] if only_wrong else scored
    cols = ["Artist", "Track", "Year", "ArtistScore", "Label", "p_best", "pred_best", "p_paper"]
    st.write(f"{len(view)} songs")
    st.dataframe(view[cols].rename(columns={"Label": "Actual (1=hit)", "p_best": "P(hit) best",
                                            "pred_best": "Predicted", "p_paper": "P(hit) paper LR"}),
                 width="stretch", hide_index=True)
    cm = pd.crosstab(scored.Label.map({1: "Actual hit", 0: "Actual non-hit"}),
                     scored.pred_best.map({1: "Predicted hit", 0: "Predicted non-hit"}))
    st.write("Confusion matrix of the best model on the test set")
    st.dataframe(cm)

with tab_custom:
    st.write("Set the features of a hypothetical song. Unset values default to the training median.")
    cols = st.columns(3)
    song = dict(DEFAULTS)
    song["ArtistScore"] = cols[0].radio("Artist already had a Hot 100 hit?", [0, 1], index=0,
                                        format_func=lambda v: "Yes" if v else "No")
    ranges = {"Loudness": (-30.0, 0.0), "Tempo": (50.0, 210.0)}
    for i, f in enumerate(AUDIO_FEATURES):
        lo, hi = ranges.get(f, (0.0, 1.0))
        song[f] = cols[(i + 1) % 3].slider(f, lo, hi, float(DEFAULTS[f]))
    df = pd.DataFrame([song]).assign(Artist="custom", Track="custom")
    s = score_songs(best, paper, df).iloc[0]
    st.metric("Best model: P(hit)", f"{s.p_best:.0%}")
    st.metric("Paper LR: P(hit)", f"{s.p_paper:.0%}")

with tab_results:
    if RESULTS_JSON.exists():
        res = json.loads(RESULTS_JSON.read_text())
        if "improvements" in res:
            steps = pd.DataFrame([{"Step": s["name"], "Grouped CV accuracy": s["cv_acc_mean"],
                                   "Test accuracy": s["test"]["accuracy"],
                                   "Test ROC AUC": s["test"].get("roc_auc")}
                                  for s in res["improvements"]["steps"]])
            st.subheader("Improvements over the paper")
            st.dataframe(steps.style.format({c: "{:.3f}" for c in steps.columns[1:]}),
                         width="stretch", hide_index=True)
    for png in sorted(FIG_DIR.glob("*.png")):
        st.image(str(png), caption=png.stem.replace("_", " "))
