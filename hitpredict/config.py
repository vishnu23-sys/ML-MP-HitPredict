"""Project-wide constants: paths, seeds, feature lists and the numbers reported in the paper."""

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"
RESULTS_JSON = RESULTS_DIR / "results.json"

# The authors' own data, published through the code link in the paper (https://tinyurl.com/yb6cvtek).
DATA_ZIP_URL = "https://drive.google.com/uc?export=download&id=1p13zKI5Q0QejLB-Q8jQBLRiJDRQNmGgD"
DATED_CSV = RAW_DIR / "complete_project_data.csv"  # 3,221 songs with release year and month
UNDATED_CSV = RAW_DIR / "complete_project_data_no_date.csv"  # 4,040 songs, superset without dates
# SHA-256 of the two CSVs as downloaded on 2026-10-05, used to verify the download.
CSV_SHA256 = {
    "complete_project_data.csv":
        "ed2e46cb2549ae7915fec758fc9d82d2484f0ae7103d124e0376a07d77c09420",
    "complete_project_data_no_date.csv":
        "5a98b89a728a9ed2b43ae8742fad7b2c4f57666db48e6c087e75f482fafe37f3",
}

AUDIO_FEATURES = [
    "Danceability", "Energy", "Speechiness", "Acousticness", "Instrumentalness",
    "Liveness", "Valence", "Loudness", "Tempo",
]
# The ten features the paper analyses (its nine audio features plus the artist score).
PAPER_FEATURES = ["ArtistScore"] + AUDIO_FEATURES
# Everything in the CSV: the paper's ten plus Key and Mode, which Spotify also returns.
ALL_FEATURES = PAPER_FEATURES + ["Key", "Mode"]
TARGET = "Label"
GROUP = "Artist"

# Feature subsets hard-coded in the authors' scripts. They were picked by forward selection
# scored on the validation set (Section 5 of the paper).
AUTHOR_SUBSETS = {
    "lr": ["ArtistScore", "Danceability", "Loudness", "Speechiness", "Acousticness",
           "Instrumentalness", "Tempo"],
    "gda": ["ArtistScore", "Danceability", "Loudness", "Speechiness", "Acousticness",
            "Instrumentalness", "Valence", "Tempo"],
    "svm": ["ArtistScore", "Danceability", "Loudness", "Speechiness", "Acousticness",
            "Instrumentalness", "Valence", "Tempo"],
    "em": ["ArtistScore", "Danceability", "Loudness", "Speechiness", "Acousticness",
           "Instrumentalness", "Tempo"],
    "ss_em": ["ArtistScore", "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
              "Instrumentalness", "Liveness", "Valence", "Tempo"],
    # project_nn.py deletes Key, Loudness, Mode and Liveness.
    "nn": ["ArtistScore", "Danceability", "Energy", "Speechiness", "Acousticness",
           "Instrumentalness", "Valence", "Tempo"],
}

# Numbers read from the paper (Georgieva, Suta and Burton, CS229 2018) and its poster.
PAPER = {
    "em_unsupervised_acc": 0.501,
    "em_semisupervised_acc": 0.469,
    "table2": {  # Figure 2 / Table: accuracy, precision, recall on training and validation
        "LR": {"train_acc": 0.735, "val_acc": 0.759, "train_prec": 0.719, "val_prec": 0.758,
               "train_rec": 0.575, "val_rec": 0.648},
        "GDA": {"train_acc": 0.735, "val_acc": 0.737, "train_prec": 0.767, "val_prec": 0.767,
                "train_rec": 0.503, "val_rec": 0.559},
        "SVM linear": {"train_acc": 0.550, "val_acc": 0.729, "train_prec": 0.783, "val_prec": 0.806,
                       "train_rec": 0.436, "val_rec": 0.490},
        "SVM RBF": {"train_acc": 0.573, "val_acc": 0.622, "train_prec": 0.749, "val_prec": 0.779,
                    "train_rec": 0.185, "val_rec": 0.173},
        "SVM polynomial": {"train_acc": 0.543, "val_acc": 0.738, "train_prec": 0.770,
                           "val_prec": 0.788, "train_rec": 0.499, "val_rec": 0.536},
        "DT": {"train_acc": 1.0, "val_acc": 0.687, "train_prec": 1.0, "val_prec": 0.658,
               "train_rec": 1.0, "val_rec": 0.657},
        "NN": {"train_acc": 0.768, "val_acc": 0.765, "train_prec": 0.875, "val_prec": 0.752,
               "train_rec": 0.600, "val_rec": 0.661},
    },
    "lr_confusion_val": {"tp": 225, "fn": 122, "fp": 72, "tn": 386},
    "bagging": {  # (model, samples per bag, bags): (train acc, val acc)
        ("SVM linear", 200, 10): (0.720, 0.729), ("SVM linear", 200, 50): (0.527, 0.544),
        ("SVM linear", 500, 10): (0.720, 0.728), ("SVM linear", 500, 50): (0.674, 0.686),
        ("SVM polynomial", 200, 10): (0.486, 0.481), ("SVM polynomial", 200, 50): (0.353, 0.349),
        ("SVM polynomial", 500, 10): (0.582, 0.576), ("SVM polynomial", 500, 50): (0.536, 0.540),
        ("DT", 200, 10): (0.165, 0.148), ("DT", 200, 50): (0.024, 0.027),
        ("DT", 500, 10): (0.210, 0.178), ("DT", 500, 50): (0.058, 0.046),
    },
    "dt_depth": {  # depth multiple of n features: (train acc, val acc)
        1: (0.840, 0.723), 1.5: (0.925, 0.709), 2: (0.990, 0.689), 10: (1.0, 0.698),
        100: (1.0, 0.703), 500: (1.0, 0.702), 1000: (1.0, 0.698), 5000: (1.0, 0.704),
    },
    "nn_best_epoch": 19000,
    "ablation": {
        "LR": [("ArtistScore", 0.729), ("Instrumentalness", 0.732), ("Danceability", 0.732),
               ("Acousticness", 0.753), ("Speechiness", 0.758), ("Loudness", 0.758),
               ("Tempo", 0.759), ("Valence", 0.757), ("Energy", 0.740), ("Liveness", 0.743)],
        "NN": [("Danceability", 0.653), ("Acousticness", 0.696), ("Speechiness", 0.730),
               ("Valence", 0.734), ("Energy", 0.749), ("ArtistScore", 0.746),
               ("Instrumentalness", 0.751), ("Tempo", 0.765), ("Liveness", 0.764),
               ("Loudness", 0.727)],
    },
    "seasonal": {  # Figure 10, accuracy on the held-out period and on the rest
        "LR": {"Summer": 0.75, "Non-summer": 0.74, "Holiday": 0.71, "Non-holiday": 0.75},
        "NN": {"Summer": 0.76, "Non-summer": 0.74, "Holiday": 0.74, "Non-holiday": 0.77},
    },
    "time_periods": {  # Figure 11, validation accuracy, read off the bar chart (about +-0.5 pt)
        "LR": {"1990-2018": 0.758, "1990-1994": 0.760, "1995-1999": 0.764, "2000-2004": 0.730,
               "2005-2009": 0.809, "2010-2014": 0.822, "2015-2018": 0.742},
        "NN": {"1990-2018": 0.764, "1990-1994": 0.787, "1995-1999": 0.797, "2000-2004": 0.734,
               "2005-2009": 0.801, "2010-2014": 0.838, "2015-2018": 0.871},
    },
}

# Plot palette (validated categorical slots 1-3 of the reference palette, light mode).
COLORS = {
    "ours": "#2a78d6",
    "paper": "#eb6834",
    "third": "#1baf7a",
    "muted": "#898781",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
