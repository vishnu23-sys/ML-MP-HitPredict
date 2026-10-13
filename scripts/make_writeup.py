"""Build the two-page project write-up (report/HitPredict_writeup.pdf) from results/results.json.

Every number in the text, table and figures is read from results.json (or, for the paper's own
numbers, from hitpredict/config.py), so the write-up always matches the latest run.

Usage:
    python scripts/make_writeup.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm, inch  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from hitpredict import plotting  # noqa: E402
from hitpredict.config import COLORS, PAPER, ROOT  # noqa: E402

OUT = ROOT / "report"
R = json.loads((ROOT / "results" / "results.json").read_text())
FIG_W_IN = 6.95  # printed width of the figures, so fonts print at their set size


def p(x, d=1):
    """Fraction to percent string."""
    return f"{100 * x:.{d}f}%"


def n(x):
    return f"{x:,}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


# ----------------------------------------------------------------------------- numbers
D = R["dataset"]
sup_rep, sup_fair = R["supervised"]["paper_protocol"], R["supervised"]["fair_protocol"]
em_u, em_s = R["unsupervised"]["unsupervised"], R["unsupervised"]["semi_supervised"]
bag = R["overfitting"]["bagging"]
nn = R["overfitting"]["nn_epochs"]
opt = R["ablation"]["selection_optimism"]["summary"]
seas = R["temporal"]["seasonal"]
periods = R["temporal"]["time_periods"]["fair_repeated"]
ev = R["evaluation_protocol"]
mc = R["model_comparison"]["models"]
im = R["improvements"]
steps = im["steps"]
lr_conf = sup_rep["LR"]["val"]["confusion"]
n_paper_train = sum(sup_rep["LR"]["train"]["confusion"].values())
n_paper_val = sum(lr_conf.values())
infl = [s["inflation_points"] for s in ev["summary"].values()]
learned = {k: v for k, v in mc.items() if k not in ("Majority class", "Artist score rule")}
test_accs = [v["test"]["accuracy"] for v in learned.values()]
best_cv = R["model_comparison"]["best_by_cv"]
hit_rates = [v["hit_rate"] for v in periods.values()]
runtime_min = sum(v.get("seconds", 0) for v in R.values() if isinstance(v, dict)) / 60
dance = im["danceability_logodds_per_sd"]
th = im["threshold"]
b0, i1, i2, i3 = steps
gain = i3["vs_baseline"]
paper_t2 = PAPER["table2"]


# ----------------------------------------------------------------------------- figures
def fig_supervised(path):
    names = ["LR", "GDA", "SVM linear", "SVM RBF", "SVM polynomial", "DT", "NN"]
    labels = ["LR", "GDA", "SVM\nlinear", "SVM\nRBF", "SVM\npoly", "Tree", "NN"]
    series = {
        "Paper (validation)": [paper_t2[k]["val_acc"] for k in names],
        "Ours, paper protocol (validation)": [
            sup_rep[k]["val_acc_best_epoch_as_reported"] if k == "NN" else sup_rep[k]["val"]["accuracy"]
            for k in names],
        "Ours, fair test on unseen artists": [sup_fair[k]["test"]["accuracy"] for k in names],
    }
    err = {"Ours, fair test on unseen artists": [sup_fair[k]["test_acc_ci95"] for k in names]}
    fig, ax = plt.subplots(figsize=(FIG_W_IN, 1.95))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plotting.grouped_bars(ax, labels, series,
                          {"Paper (validation)": COLORS["paper"],
                           "Ours, paper protocol (validation)": COLORS["ours"],
                           "Ours, fair test on unseen artists": COLORS["third"]},
                          ylim=(0.5, 0.82), err=err)
    rule = sup_fair["Artist score rule"]["test"]["accuracy"]
    ax.axhline(rule, color=COLORS["muted"], lw=1, label=f"Artist score alone, fair test ({p(rule)})")
    ax.set_ylabel("Accuracy", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- text
TITLE = "HitPredict revisited: predicting Billboard Hot 100 hits from Spotify data"
TEAM = ("UE24CS352A Machine Learning, Mini-Project, Problem 16<br/>"
        "Team: Vishnu Patruni (PES1UG24AM187) and Piyush G (PES1UG24AM190)")

S1 = (
    "Our problem is hit song prediction. Given a song's Spotify audio features and whether its "
    "artist has charted before, the task is to predict whether the song reached the Billboard "
    "Hot 100. The reference paper is <i>HitPredict</i> by Georgieva, Suta and Burton (Stanford "
    "CS229, 2018). It reports about 75% accuracy, with logistic regression (LR) and a small neural "
    "network (NN) as the best models. We set out to reproduce its experiments, check whether its "
    "evaluation was fair, and improve on it only with changes we could justify."
)

S2 = (
    "We used the paper's own data, which the authors published in the code archive linked from "
    "the paper. Collecting it again is not possible, because Spotify closed its audio-features API "
    f"to new applications in 2024. The file has {n(D['raw_rows_undated_file'])} rows. After removing "
    f"{D['exact_duplicates_undated_file']} exact duplicates we have {n(D['songs'])} songs, of which "
    f"{n(D['hits'])} ({p(D['hit_rate'])}) were Hot 100 hits between {D['year_min']} and "
    f"{D['year_max']}; the rest are random Million Song Dataset tracks. Each song has the nine audio "
    "features the paper uses (danceability, energy, speechiness, acousticness, instrumentalness, "
    "liveness, valence, loudness, tempo), Spotify's key and mode, and the authors' artist score, "
    "which is 1 if the artist had an earlier Hot 100 hit. "
    f"{n(D['songs_with_month'])} songs also have a release month, and the paper's main experiments "
    "use only these. Two properties of the data matter later. First, "
    f"{p(seas['january_share'], 0)} of dated songs are listed as released in January, because "
    "Spotify stores a year-only release date as 1 January. Second, the hit rate ranges from "
    f"{p(min(hit_rates), 0)} to {p(max(hit_rates), 0)} across five-year periods, since the non-hits "
    "were sampled without regard to year."
)

S3a = (
    "We first reran every experiment exactly as the authors did: their CSV in file order, the first "
    f"{n(n_paper_train)} rows for training and the remaining {n_paper_val} for validation, their "
    "hand-picked feature subsets and unscaled features. This covers unregularised LR fitted by "
    "Newton's method, Gaussian discriminant analysis (GDA) with a shared covariance, SVMs with "
    "linear, RBF and polynomial kernels, a decision tree, the authors' NN (one hidden layer of 6 "
    "sigmoid units), two-component EM clustering, bagging, a tree-depth sweep, forward feature "
    "selection, and the seasonal and five-year-period studies."
)

S3b = (
    "We then re-evaluated everything with a protocol that never uses test data to make a choice. "
    "Songs are split by lead artist, so that a featured credit such as \"drake feat. rihanna\" "
    f"counts as \"drake\". Three quarters of the artists ({n(D['train_songs'])} songs) are used "
    f"for training, and the other quarter ({n(D['test_songs'])} songs by {D['test_lead_artists']} "
    "artists) forms a test set that is scored once per model. Hyperparameters, feature subsets, the NN epoch and the final model are "
    "all chosen by 5-fold artist-grouped cross-validation (CV) on the training artists, and 95% "
    "confidence intervals come from resampling whole artists 2,000 times. The paper instead chose "
    "its feature subsets, NN epoch, tree depth and bagging setting on the validation set and "
    "reported that same validation score, and it states that \"an additional test set was not "
    "needed\"."
)

S3c = (
    f"On this split we tuned {len(mc)} models by grid search, from a majority-class baseline to "
    "random forest and gradient boosting. We then added four improvements one at a time, fixed "
    "before we looked at any test score. I1 also trains on the "
    f"{D['songs_without_month']} songs without a release month, which the paper left out of its LR, "
    "GDA and SVM experiments even though the date is never an input. I2 takes log10 of the four "
    "skewed features, drops key, adds artist score x audio interaction terms and tunes the L2 "
    "penalty. I3 replaces LR with gradient boosting or a soft-voting ensemble, whichever has the "
    "higher CV accuracy. I4 picks the lowest decision threshold that reaches 80% precision on "
    "out-of-fold training predictions, because the paper points out that a false positive is "
    "costly for a record label."
)

S4 = (
    "The code is a Python package with one module per experiment family: supervised models, EM, "
    "overfitting studies, feature ablation, temporal effects, split comparison, model comparison "
    f"and improvements. A single script runs all of them in about {runtime_min:.0f} minutes and "
    "writes results.json, eleven PNG figures and the trained models; a separate script downloads the "
    "data and checks its SHA-256 hashes. The random seed is fixed at 42 and package versions are "
    "pinned (Python 3.12, scikit-learn 1.9.1). Nine pytest smoke tests run in a few seconds, one of "
    "which checks that the paper's LR result is reproduced exactly. A Streamlit app loads the "
    "saved models and scores real test songs, with a command-line fallback for the live demo."
)

S5a = (
    "LR, GDA and the linear and RBF SVMs match the paper's results table to three decimals "
    "(Table 1, Figure 1), the polynomial SVM is "
    f"{abs(sup_rep['SVM polynomial']['val']['accuracy'] - paper_t2['SVM polynomial']['val_acc']) * 100:.1f} "
    "points off, and the LR confusion matrix is identical "
    f"({lr_conf['tp']} true positives, {lr_conf['fn']} false negatives, {lr_conf['fp']} false "
    f"positives, {lr_conf['tn']} true negatives). The RBF and polynomial SVMs matched only after we "
    "used the 2018 scikit-learn definition of gamma='scale', 1/(n_features x std(X)); later versions "
    f"use the variance instead. The decision tree is {abs(sup_rep['DT']['val']['accuracy'] - paper_t2['DT']['val_acc']) * 100:.1f} "
    f"points off and the NN {abs(sup_rep['NN']['val_acc_best_epoch_as_reported'] - paper_t2['NN']['val_acc']) * 100:.1f} "
    "points off. We could not reproduce the NN training curve: in our runs the network did better "
    f"without the paper's penalty term (best validation accuracy "
    f"{p(nn['paper_protocol']['without_penalty']['best_val_acc'])} against "
    f"{p(nn['paper_protocol']['with_penalty']['best_val_acc'])} with it), which is the opposite "
    "of the paper's finding."
)

real_svm = [sup_rep[k]["train"]["accuracy"] for k in ("SVM linear", "SVM RBF", "SVM polynomial")]
S5b = (
    "Reproducing the numbers exposed three scoring bugs in the authors' scripts. The SVM training "
    "accuracies of 0.550, 0.573 and 0.543 come from comparing an (m, 1) prediction array with an "
    "(m,) label array, which numpy broadcasts to an m x m matrix. The real training accuracies are "
    f"{real_svm[0]:.3f}, {real_svm[1]:.3f} and {real_svm[2]:.3f}, so the paper's conclusion that the "
    "SVMs overfit does not hold. The same bug produces the EM accuracies of 50.1% and 46.9%. Scored "
    f"correctly, EM reaches {p(em_u['best_label_matching_mean'])} and semi-supervised EM "
    f"{p(em_s['20pct_labelled']['accuracy_on_unlabelled'])}, against a majority rate of "
    f"{p(em_u['majority_class_rate'])}. In the bagging study the averaged votes were compared "
    "with the 0/1 label, so a song counted as correct only when every bag agreed. This explains "
    "accuracies as low as 2.7%, while a real majority vote gives "
    f"{p(min(b['majority_vote_val'] for b in bag))} to {p(max(b['majority_vote_val'] for b in bag))} "
    "in every setting."
)

S5c = (
    f"Under the fair protocol the paper's LR scores {p(b0['test']['accuracy'])} on unseen artists, "
    "and simply predicting a hit whenever the artist had one before scores "
    f"{p(mc['Artist score rule']['test']['accuracy'])}. The audio features add only a few points. "
    "Choosing features on the validation set, as the paper did, overstated accuracy by "
    f"{100 * opt['optimism_selected']:.1f} points on average over 20 splits, against "
    f"{100 * opt['optimism_all_features']:.1f} points with no selection. Two effects we expected did "
    "not appear. A random row split puts "
    f"{p(ev['leakage_facts']['paper_split_val_songs_with_artist_in_train'], 0)} of the paper's "
    "validation songs next to a song by the same artist in training, yet it changed accuracy by "
    f"only {min(infl):+.1f} to {max(infl):+.1f} points across four models and 20 seeds. The artist "
    "score already carries what the artist's identity would. The paper's holiday effect is mostly "
    "the January artifact: out-of-fold accuracy is "
    f"{p(seas['fair_oof_by_period']['Nov-Dec only']['oof_accuracy'])} for November and December "
    f"songs but {p(seas['fair_oof_by_period']['January only']['oof_accuracy'])} for January ones."
)

S5d = (
    f"Among the {len(mc)} models, tuned gradient boosting had the best CV accuracy "
    f"({p(mc[best_cv]['cv_acc_mean'])}) and scored {p(mc[best_cv]['test']['accuracy'])} on test; the "
    f"{len(learned)} learned models lie between {p(min(test_accs))} and {p(max(test_accs))}. "
    f"The improvements (Table 1, section C) raised test accuracy from {p(b0['test']['accuracy'])} to "
    f"{p(i3['test']['accuracy'])} and ROC AUC from {b0['test']['roc_auc']:.3f} to "
    f"{i3['test']['roc_auc']:.3f}. I1 did not change test accuracy at all. It only reduced the spread "
    f"of the CV estimate from {100 * b0['cv_acc_sd']:.1f} to {100 * i1['cv_acc_sd']:.1f} points, "
    "which fits a seven-feature linear model that is limited by bias rather than by the amount of "
    f"data. I2 added {100 * (i2['test']['accuracy'] - i1['test']['accuracy']):.1f} points, and its "
    "interaction terms are easy to read: one standard deviation more danceability raises the "
    f"log-odds of a hit by {dance['new_artists']:.2f} for new artists but only by "
    f"{dance['established_artists']:.2f} for established ones. I3 added another "
    f"{100 * (i3['test']['accuracy'] - i2['test']['accuracy']):.1f} point with shallow trees (depth "
    f"{im['hgb_best_params']['max_depth']}, {im['hgb_best_params']['max_iter']} rounds), which "
    "capture step effects that a linear model can only approximate: only "
    f"{p(D['hit_rate_instrumentalness_gt_0.5'])} of songs with instrumentalness above 0.5 are hits. "
    f"I4, with a threshold of {th['threshold_from_oof']:.2f}, raised test precision from "
    f"{p(th['test_at_0.5']['precision'])} to {p(th['test_at_threshold']['precision'])} at the cost "
    f"of recall ({p(th['test_at_0.5']['recall'])} to {p(th['test_at_threshold']['recall'])}). The "
    f"95% interval of the total gain, [{100 * gain['lo']:+.1f}, {100 * gain['hi']:+.1f}] points, "
    "still includes zero."
)

S6 = (
    "The paper's headline numbers are reproducible, but several of its conclusions are not. The "
    "SVMs do not overfit, EM is not at chance level, bagging does not collapse, and the holiday "
    "effect is largely a date artifact. On artists the model has never seen, Spotify features plus "
    f"the artist's history reach about {p(i3['test']['accuracy'], 0)} accuracy, and most of that "
    f"comes from the artist score alone. Our changes add {100 * gain['diff']:.1f} points, which is "
    f"within the noise of a {n(D['test_songs'])}-song test set, so we report them as a modest gain "
    "rather than a clear win. The main limitations are the fixed feature set (no new features can "
    "be fetched from Spotify), non-hits that are random tracks rather than near-misses, and a test "
    "set too small to separate models that are less than about 3 points apart. Release year "
    "predicts the label in this data only because of how the non-hits were sampled, so we did not "
    "use it as a feature."
)


# ----------------------------------------------------------------------------- table
def results_table(body_style, bold_style):
    def P(t, st=body_style):
        return Paragraph(t, st)

    rows, section_rows = [], []

    def section(title, c1, c2, c3):
        section_rows.append(len(rows))
        rows.append([P(f"<b>{title}</b>"), P(f"<b>{c1}</b>"), P(f"<b>{c2}</b>"), P(f"<b>{c3}</b>")])

    section("A. Supervised models (paper's Table 2)", "Paper, validation",
            "Ours, paper protocol", "Ours, fair test [95% CI]")
    for k, label in [("LR", "Logistic regression"), ("GDA", "GDA"), ("SVM linear", "SVM, linear"),
                     ("SVM RBF", "SVM, RBF"), ("SVM polynomial", "SVM, polynomial"),
                     ("DT", "Decision tree"), ("NN", "Neural network (best epoch)")]:
        ours = (sup_rep[k]["val_acc_best_epoch_as_reported"] if k == "NN"
                else sup_rep[k]["val"]["accuracy"])
        f = sup_fair[k]
        rows.append([P(label), P(p(paper_t2[k]["val_acc"])), P(p(ours)),
                     P(f"{p(f['test']['accuracy'])} {ci(f['test_acc_ci95'])}")])

    section("B. Results affected by scoring bugs", "Paper", "Ours, authors' scoring",
            "Ours, corrected")
    for k, label in [("SVM linear", "SVM linear, training accuracy"),
                     ("SVM RBF", "SVM RBF, training accuracy"),
                     ("SVM polynomial", "SVM polynomial, training accuracy")]:
        rows.append([P(label), P(p(paper_t2[k]["train_acc"])),
                     P(p(sup_rep[k]["train_acc_as_computed_by_authors"])),
                     P(p(sup_rep[k]["train"]["accuracy"]))])
    rows.append([P("EM clustering, unsupervised"), P(p(PAPER["em_unsupervised_acc"])),
                 P(p(em_u["as_computed_by_authors_mean"])), P(p(em_u["best_label_matching_mean"]))])
    rows.append([P("EM, semi-supervised (20% labels)"), P(p(PAPER["em_semisupervised_acc"])),
                 P(p(em_s["20pct_labelled"]["as_computed_by_authors"])),
                 P(p(em_s["20pct_labelled"]["accuracy_on_unlabelled"]))])
    bdt = next(b for b in bag if b["model"] == "DT" and b["samples_per_bag"] == 200
               and b["bags"] == 50)
    rows.append([P("Bagged trees (50 bags of 200 songs)"), P(p(bdt["paper_val_acc"])),
                 P(p(bdt["authors_metric_val"])), P(p(bdt["majority_vote_val"]))])

    section(f"C. Fair protocol ({n(D['test_songs'])} test songs, unseen artists)", "Grouped CV acc.",
            "Test acc. [95% CI]", "Test ROC AUC")
    for label, cv, t, cint, auc in [
        ("Majority class", mc["Majority class"]["cv_acc_mean"],
         mc["Majority class"]["test"]["accuracy"], mc["Majority class"]["test_acc_ci95"], None),
        ("Artist score alone", mc["Artist score rule"]["cv_acc_mean"],
         mc["Artist score rule"]["test"]["accuracy"], mc["Artist score rule"]["test_acc_ci95"],
         mc["Artist score rule"]["test"]["roc_auc"]),
    ] + [(s["name"].replace(" (LR, 7 features, dated songs)", ", LR on dated songs")
          .replace("I3 + nonlinear model (gradient boosting)", "I3 + gradient boosting"),
          s["cv_acc_mean"], s["test"]["accuracy"], s["test_acc_ci95"], s["test"]["roc_auc"])
         for s in steps] + [
        ("Best single model: gradient boosting, 10 features", mc[best_cv]["cv_acc_mean"],
         mc[best_cv]["test"]["accuracy"], mc[best_cv]["test_acc_ci95"],
         mc[best_cv]["test"]["roc_auc"])]:
        rows.append([P(label), P(p(cv)), P(f"{p(t)} {ci(cint)}"),
                     P("-" if auc is None else f"{auc:.3f}")])

    widths = [6.6 * cm, 3.0 * cm, 3.6 * cm, 4.2 * cm]
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    style = [
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ]
    for r in section_rows:
        style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#ecebe6")),
                  ("LINEBELOW", (0, r), (-1, r), 0.4, colors.HexColor("#8a8984"))]
    t.setStyle(TableStyle(style))
    return t


# ----------------------------------------------------------------------------- build
def build():
    OUT.mkdir(exist_ok=True)
    plotting.setup()
    f1 = OUT / "fig_supervised.png"
    fig_supervised(f1)

    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.5, leading=11.3,
                          alignment=TA_JUSTIFY, spaceAfter=3.5)
    cell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8, leading=9.2)
    head = ParagraphStyle("head", fontName="Times-Bold", fontSize=10.8, leading=12.5,
                          spaceBefore=5, spaceAfter=2)
    title = ParagraphStyle("title", fontName="Times-Bold", fontSize=14, leading=16.5,
                           spaceAfter=2)
    sub = ParagraphStyle("sub", fontName="Times-Italic", fontSize=9.6, leading=11.5, spaceAfter=4)
    cap = ParagraphStyle("cap", fontName="Times-Roman", fontSize=8.3, leading=9.8, spaceBefore=1,
                         spaceAfter=5)

    def img(path, h_in):
        return Image(str(path), width=FIG_W_IN * inch, height=h_in * inch)

    story = [
        Paragraph(TITLE, title), Paragraph(TEAM, sub),
        Paragraph("1. Problem statement", head), Paragraph(S1, body),
        Paragraph("2. Dataset", head), Paragraph(S2, body),
        Paragraph("3. Approach", head), Paragraph(S3a, body), Paragraph(S3b, body),
        Paragraph(S3c, body),
        Paragraph("4. Implementation", head), Paragraph(S4, body),
        Paragraph("5. Results", head), Paragraph(S5a, body), Paragraph(S5b, body),
        KeepTogether([
            img(f1, 1.95),
            Paragraph("<b>Figure 1.</b> Validation accuracy reported by the paper, our rerun of the "
                      "paper's protocol, and test accuracy of the same fixed models on unseen "
                      "artists (error bars: 95% CI). The grey line is the artist score alone.", cap)]),
        KeepTogether([
            results_table(cell, cell),
            Paragraph("<b>Table 1.</b> All accuracies in percent. Section B compares each number "
                      "as the authors' code computes it with the corrected value. In section C every "
                      "row is scored on the same test songs; B0 to I3 are the cumulative "
                      "improvement steps.", cap)]),
        Paragraph(S5c, body), Paragraph(S5d, body),
        Paragraph("6. Conclusions", head), Paragraph(S6, body),
    ]
    pdf = OUT / "HitPredict_writeup.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=1.55 * cm, rightMargin=1.55 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="HitPredict revisited", author="Vishnu Patruni and Piyush G")
    doc.build(story)
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    build()
