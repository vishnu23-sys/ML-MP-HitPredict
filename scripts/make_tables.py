"""Print the README results tables (Markdown) from results/results.json.

Usage:
    python scripts/make_tables.py > results/tables.md
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitpredict.config import PAPER, RESULTS_JSON  # noqa: E402


def pct(v, d=1):
    return "n/a" if v is None else f"{100 * v:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    R = json.loads(RESULTS_JSON.read_text())
    parts = []

    if "supervised" in R:
        rep, fair = R["supervised"]["paper_protocol"], R["supervised"]["fair_protocol"]
        rows = []
        for n in ["LR", "GDA", "SVM linear", "SVM RBF", "SVM polynomial", "DT", "NN"]:
            p, o = PAPER["table2"][n], rep[n]
            our_val = o["val_acc_best_epoch_as_reported"] if n == "NN" else o["val"]["accuracy"]
            our_tr = o.get("train_acc_as_computed_by_authors", o["train"]["accuracy"])
            ci = fair[n]["test_acc_ci95"]
            rows.append([n, pct(p["train_acc"]), pct(our_tr), pct(p["val_acc"]), pct(our_val),
                         pct(p["val_prec"]), pct(o["val"]["precision"]), pct(p["val_rec"]),
                         pct(o["val"]["recall"]),
                         f"{pct(fair[n]['test']['accuracy'])} [{pct(ci[0])}, {pct(ci[1])}]"])
        parts.append("### Table 2 (supervised models): paper vs ours\n\n" + table(
            ["Model", "Paper train acc", "Ours train acc*", "Paper val acc", "Ours val acc",
             "Paper val prec", "Ours val prec", "Paper val rec", "Ours val rec",
             "Ours, fair test acc [95% CI]"], rows))

    if "unsupervised" in R:
        u, s = R["unsupervised"]["unsupervised"], R["unsupervised"]["semi_supervised"]
        parts.append("### EM clustering\n\n" + table(
            ["Experiment", "Paper", "Authors' metric, ours", "Real accuracy, ours"],
            [["Unsupervised EM (2 Gaussians)", pct(PAPER["em_unsupervised_acc"]),
              pct(u["as_computed_by_authors_mean"]), pct(u["best_label_matching_mean"])],
             ["Semi-supervised EM, 5% labels (authors' code)", "n/a",
              pct(s["5pct_labelled"]["as_computed_by_authors"]),
              pct(s["5pct_labelled"]["accuracy_on_unlabelled"])],
             ["Semi-supervised EM, 20% labels (paper text)", pct(PAPER["em_semisupervised_acc"]),
              pct(s["20pct_labelled"]["as_computed_by_authors"]),
              pct(s["20pct_labelled"]["accuracy_on_unlabelled"])],
             ["Majority class", "", "", pct(u["majority_class_rate"])]]))

    if "overfitting" in R:
        o = R["overfitting"]
        rows = [[b["model"], f"{b['samples_per_bag']} x {b['bags']}", pct(b["paper_val_acc"]),
                 pct(b["authors_metric_val"]), pct(b["majority_vote_val"])] for b in o["bagging"]]
        parts.append("### Bagging\n\n" + table(
            ["Model", "Songs per bag x bags", "Paper val acc", "Authors' metric, ours",
             "Real majority vote, ours"], rows))
        rows = [[f"{d['depth_multiple']}n ({d['max_depth']})", pct(d["paper_train_acc"]),
                 pct(d["train_acc"]), pct(d["paper_val_acc"]), pct(d["val_acc"])]
                for d in o["tree_depth"]["paper_protocol"]]
        t = o["tree_depth"]
        parts.append("### Decision-tree depth\n\n" + table(
            ["Max depth", "Paper train", "Ours train", "Paper val", "Ours val"], rows) +
            f"\n\nFair protocol: depth {t['fair_chosen_depth']} chosen by grouped CV, test accuracy "
            f"{pct(t['fair_test']['accuracy'])}%.")
        nn = o["nn_epochs"]
        pp, f = nn["paper_protocol"], nn["fair"]
        parts.append("### Neural network epochs\n\n" + table(
            ["Setting", "Best epoch", "Best val acc", "Val acc at last epoch"],
            [["Paper", PAPER["nn_best_epoch"], pct(PAPER["table2"]["NN"]["val_acc"]), ""],
             ["Ours, with penalty", pp["with_penalty"]["best_epoch"],
              pct(pp["with_penalty"]["best_val_acc"]), pct(pp["with_penalty"]["final_val_acc"])],
             ["Ours, without penalty", pp["without_penalty"]["best_epoch"],
              pct(pp["without_penalty"]["best_val_acc"]),
              pct(pp["without_penalty"]["final_val_acc"])]]) +
            f"\n\nFair protocol: epoch {f['epoch_chosen_on_inner_validation']} chosen on an inner "
            f"validation split, test accuracy {pct(f['test_acc'])}% (the best any epoch reaches on "
            f"the test set, i.e. what peeking would report: "
            f"{pct(f['max_test_acc_over_epochs_if_peeking'])}%).")

    if "ablation" in R:
        a = R["ablation"]
        rows = []
        for i in range(10):
            pl, ol = PAPER["ablation"]["LR"][i], a["paper_protocol"]["LR"][i]
            pn, on = PAPER["ablation"]["NN"][i], a["paper_protocol"]["NN"][i]
            rows.append([i + 1, f"{pl[0]} {pct(pl[1])}", f"{ol[0]} {pct(ol[1])}",
                         f"{pn[0]} {pct(pn[1])}", f"{on[0]} {pct(on[1])}"])
        s = a["selection_optimism"]["summary"]
        fs = a["fair_selection"]
        parts.append("### Feature ablation (forward selection on validation)\n\n" + table(
            ["Step", "Paper LR", "Ours LR", "Paper NN", "Ours NN"], rows) +
            f"\n\nFair selection with grouped CV picked {len(fs['selected'])} features "
            f"({', '.join(fs['selected'])}); test accuracy {pct(fs['test_acc_selected_subset'])}% "
            f"vs {pct(fs['test_acc_all_10_features'])}% with all 10 features. Over 20 random "
            f"splits, choosing features on the validation set overstates accuracy by "
            f"{pct(s['optimism_selected'])} points on average (validation "
            f"{pct(s['selected_val_acc'])}% vs fresh test {pct(s['selected_test_acc'])}%), "
            f"against {pct(s['optimism_all_features'])} points when no selection is done.")

    if "temporal" in R:
        t = R["temporal"]
        sp = t["seasonal"]["paper_protocol"]
        fair = t["seasonal"]["fair_oof_by_period"]
        rows = []
        for season, ins, outs in [("Summer", "Summer (Jun-Aug)", "Non-summer"),
                                  ("Holiday", "Holiday (Nov-Jan)", "Non-holiday")]:
            rows.append([season, pct(PAPER["seasonal"]["LR"][season], 0),
                         pct(sp[season]["LR_test_inside"]),
                         pct(PAPER["seasonal"]["LR"][f"Non-{season.lower()}"], 0),
                         pct(sp[season]["LR_train_outside"]),
                         f"{pct(fair[ins]['oof_accuracy'])} vs {pct(fair[outs]['oof_accuracy'])}",
                         f"{pct(fair[ins]['hit_rate'], 0)} vs {pct(fair[outs]['hit_rate'], 0)}"])
        parts.append("### Seasonal effects (logistic regression)\n\n" + table(
            ["Period", "Paper in-period", "Ours in-period", "Paper out-of-period",
             "Ours out-of-period", "Fair OOF acc, in vs out", "Hit rate, in vs out"], rows) +
            f"\n\n{pct(t['seasonal']['january_share'], 0)}% of dated songs carry a January release "
            f"month. Nov-Dec only: OOF accuracy {pct(fair['Nov-Dec only']['oof_accuracy'])}% "
            f"(n={fair['Nov-Dec only']['n']}); January only: "
            f"{pct(fair['January only']['oof_accuracy'])}% (n={fair['January only']['n']}).")
        tp, fr = t["time_periods"]["paper_protocol"], t["time_periods"]["fair_repeated"]
        rows = []
        for n in tp:
            f = fr.get(n)
            rows.append([n, tp[n]["n"], pct(PAPER["time_periods"]["LR"][n]), pct(tp[n]["LR_val"]),
                         pct(PAPER["time_periods"]["NN"][n]), pct(tp[n]["NN_val_best_epoch"]),
                         f"{pct(f['period_model_mean'])} +- {pct(f['period_model_sd'])}" if f else "",
                         pct(f["pooled_model_mean"]) if f else "",
                         pct(f["majority_mean"]) if f else "",
                         pct(f["hit_rate"], 0) if f else ""])
        parts.append("### Five-year periods\n\n" + table(
            ["Period", "Songs", "Paper LR", "Ours LR", "Paper NN", "Ours NN",
             "Fair: period model (30 splits)", "Fair: pooled model", "Majority baseline",
             "Hit rate"], rows))

    if "evaluation_protocol" in R:
        e = R["evaluation_protocol"]
        rows = [[n, f"{pct(s['random_mean'])} +- {pct(s['random_sd'])}",
                 f"{pct(s['grouped_mean'])} +- {pct(s['grouped_sd'])}",
                 f"{s['inflation_points']:+.1f}", pct(s["temporal"])]
                for n, s in e["summary"].items()]
        ti = e["temporal_split"]
        parts.append("### Random vs artist-grouped split (20 seeds)\n\n" + table(
            ["Model", "Random split", "Artist-grouped split", "Inflation (points)",
             "Temporal split"], rows) +
            f"\n\nTemporal split: {ti['n_train']} training songs (<=2006), {ti['n_test']} test "
            f"songs (>=2007); hit rate {pct(ti['train_hit_rate'], 0)}% vs "
            f"{pct(ti['test_hit_rate'], 0)}%, majority baseline on test "
            f"{pct(ti['test_majority_baseline'])}%.")

    if "model_comparison" in R:
        m = R["model_comparison"]["models"]
        rows = []
        for n in sorted(m, key=lambda k: -m[k]["cv_acc_mean"]):
            r = m[n]
            ci = r["test_acc_ci95"]
            rows.append([n, f"{pct(r['cv_acc_mean'])} +- {pct(r['cv_acc_sd'])}",
                         f"{pct(r['test']['accuracy'])} [{pct(ci[0])}, {pct(ci[1])}]",
                         pct(r["test"]["precision"]), pct(r["test"]["recall"]),
                         f"{r['test'].get('roc_auc', float('nan')):.3f}",
                         ", ".join(f"{k.replace('clf__', '')}={v}"
                                   for k, v in r["best_params"].items()) or "-"])
        parts.append("### Model comparison (10 paper features, tuned by grouped CV)\n\n" + table(
            ["Model", "Grouped CV acc", "Test acc [95% CI]", "Test precision", "Test recall",
             "Test ROC AUC", "Chosen hyperparameters"], rows) +
            f"\n\nBest by CV: **{R['model_comparison']['best_by_cv']}**.")

    if "improvements" in R:
        im = R["improvements"]
        rows = []
        for s in im["steps"]:
            ci = s["test_acc_ci95"]
            vb = s["vs_baseline"]
            rows.append([s["name"], f"{pct(s['cv_acc_mean'])} +- {pct(s['cv_acc_sd'])}",
                         f"{pct(s['test']['accuracy'])} [{pct(ci[0])}, {pct(ci[1])}]",
                         f"{s['test'].get('roc_auc', float('nan')):.3f}",
                         f"{100 * vb['diff']:+.1f} [{100 * vb['lo']:+.1f}, {100 * vb['hi']:+.1f}]"])
        th = im["threshold"]
        parts.append("### Improvements (cumulative, same test artists)\n\n" + table(
            ["Step", "Grouped CV acc", "Test acc [95% CI]", "Test ROC AUC",
             "vs B0, points [95% CI]"], rows) +
            f"\n\nI3 candidates (grouped CV): " + ", ".join(
                f"{k} {pct(v['cv_acc_mean'])}%" for k, v in im["i3_candidates"].items()) +
            f". Deployed model: **{im['final_model']}**.\n\n"
            f"I4 threshold {th['threshold_from_oof']:.2f}: test precision "
            f"{pct(th['test_at_threshold']['precision'])}% and recall "
            f"{pct(th['test_at_threshold']['recall'])}%, vs "
            f"{pct(th['test_at_0.5']['precision'])}% and {pct(th['test_at_0.5']['recall'])}% at 0.5.")

    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
