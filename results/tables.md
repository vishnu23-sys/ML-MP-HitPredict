### Table 2 (supervised models): paper vs ours

| Model | Paper train acc | Ours train acc* | Paper val acc | Ours val acc | Paper val prec | Ours val prec | Paper val rec | Ours val rec | Ours, fair test acc [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| LR | 73.5 | 73.5 | 75.9 | 75.9 | 75.8 | 75.8 | 64.8 | 64.8 | 73.1 [70.2, 76.2] |
| GDA | 73.5 | 73.5 | 73.7 | 73.7 | 76.7 | 76.7 | 55.9 | 55.9 | 73.8 [70.4, 77.1] |
| SVM linear | 55.0 | 55.0 | 72.9 | 72.9 | 80.6 | 80.6 | 49.0 | 49.0 | 72.1 [68.7, 75.5] |
| SVM RBF | 57.3 | 57.3 | 62.2 | 62.2 | 77.9 | 77.9 | 17.3 | 17.3 | 64.2 [60.2, 68.3] |
| SVM polynomial | 54.3 | 54.3 | 73.8 | 73.7 | 78.8 | 78.5 | 53.6 | 53.6 | 73.1 [69.8, 76.3] |
| DT | 100.0 | 100.0 | 68.7 | 70.1 | 65.8 | 65.2 | 65.7 | 65.4 | 67.9 [64.9, 71.1] |
| NN | 76.8 | 74.9 | 76.5 | 74.9 | 75.2 | 59.1 | 66.1 | 83.1 | 73.4 [70.1, 76.8] |

### EM clustering

| Experiment | Paper | Authors' metric, ours | Real accuracy, ours |
|---|---|---|---|
| Unsupervised EM (2 Gaussians) | 50.1 | 50.0 | 62.6 |
| Semi-supervised EM, 5% labels (authors' code) | n/a | 47.8 | 65.5 |
| Semi-supervised EM, 20% labels (paper text) | 46.9 | 48.2 | 68.5 |
| Majority class |  |  | 58.6 |

### Bagging

| Model | Songs per bag x bags | Paper val acc | Authors' metric, ours | Real majority vote, ours |
|---|---|---|---|---|
| SVM linear | 200 x 10 | 72.9 | 72.9 | 72.9 |
| SVM linear | 200 x 50 | 54.4 | 65.7 | 72.9 |
| SVM linear | 500 x 10 | 72.8 | 72.7 | 72.9 |
| SVM linear | 500 x 50 | 68.6 | 70.6 | 72.9 |
| SVM polynomial | 200 x 10 | 48.1 | 49.1 | 72.9 |
| SVM polynomial | 200 x 50 | 34.9 | 35.0 | 73.0 |
| SVM polynomial | 500 x 10 | 57.6 | 58.1 | 72.3 |
| SVM polynomial | 500 x 50 | 54.0 | 52.2 | 73.4 |
| DT | 200 x 10 | 14.8 | 15.2 | 75.4 |
| DT | 200 x 50 | 2.7 | 1.5 | 75.5 |
| DT | 500 x 10 | 17.8 | 19.0 | 72.3 |
| DT | 500 x 50 | 4.6 | 4.2 | 74.9 |

### Decision-tree depth

| Max depth | Paper train | Ours train | Paper val | Ours val |
|---|---|---|---|---|
| 1n (8) | 84.0 | 84.1 | 72.3 | 72.4 |
| 1.5n (12) | 92.5 | 92.4 | 70.9 | 70.9 |
| 2n (16) | 99.0 | 99.0 | 68.9 | 70.7 |
| 10n (80) | 100.0 | 100.0 | 69.8 | 70.1 |
| 100n (800) | 100.0 | 100.0 | 70.3 | 70.1 |
| 500n (4000) | 100.0 | 100.0 | 70.2 | 70.1 |
| 1000n (8000) | 100.0 | 100.0 | 69.8 | 70.1 |
| 5000n (40000) | 100.0 | 100.0 | 70.4 | 70.1 |

Fair protocol: depth 1 chosen by grouped CV, test accuracy 72.1%.

### Neural network epochs

| Setting | Best epoch | Best val acc | Val acc at last epoch |
|---|---|---|---|
| Paper | 19000 | 76.5 |  |
| Ours, with penalty | 1700 | 74.9 | 70.7 |
| Ours, without penalty | 20600 | 76.9 | 76.7 |

Fair protocol: epoch 100 chosen on an inner validation split, test accuracy 73.6% (the best any epoch reaches on the test set, i.e. what peeking would report: 74.6%).

### Feature ablation (forward selection on validation)

| Step | Paper LR | Ours LR | Paper NN | Ours NN |
|---|---|---|---|---|
| 1 | ArtistScore 72.9 | ArtistScore 72.9 | Danceability 65.3 | ArtistScore 73.3 |
| 2 | Instrumentalness 73.2 | Instrumentalness 73.2 | Acousticness 69.6 | Danceability 73.5 |
| 3 | Danceability 73.2 | Danceability 73.2 | Speechiness 73.0 | Loudness 75.4 |
| 4 | Acousticness 75.3 | Acousticness 75.3 | Valence 73.4 | Speechiness 76.0 |
| 5 | Speechiness 75.8 | Speechiness 75.8 | Energy 74.9 | Liveness 75.7 |
| 6 | Loudness 75.8 | Energy 75.8 | ArtistScore 74.6 | Instrumentalness 76.0 |
| 7 | Tempo 75.9 | Liveness 76.1 | Instrumentalness 75.1 | Acousticness 76.6 |
| 8 | Valence 75.7 | Valence 76.0 | Tempo 76.5 | Tempo 76.0 |
| 9 | Energy 74.0 | Tempo 76.5 | Liveness 76.4 | Valence 76.5 |
| 10 | Liveness 74.3 | Loudness 74.9 | Loudness 72.7 | Energy 70.2 |

Fair selection with grouped CV picked 10 features (ArtistScore, Energy, Liveness, Speechiness, Tempo, Loudness, Danceability, Instrumentalness, Valence, Acousticness); test accuracy 74.9% vs 74.9% with all 10 features. Over 20 random splits, choosing features on the validation set overstates accuracy by 1.7 points on average (validation 76.0% vs fresh test 74.3%), against 1.0 points when no selection is done.

### Seasonal effects (logistic regression)

| Period | Paper in-period | Ours in-period | Paper out-of-period | Ours out-of-period | Fair OOF acc, in vs out | Hit rate, in vs out |
|---|---|---|---|---|---|---|
| Summer | 75 | 74.7 | 74 | 74.1 | 75.0 vs 74.3 | 39 vs 42 |
| Holiday | 71 | 70.9 | 75 | 75.2 | 72.9 vs 75.6 | 48 vs 37 |

34% of dated songs carry a January release month. Nov-Dec only: OOF accuracy 78.4% (n=324); January only: 71.3% (n=1103).

### Five-year periods

| Period | Songs | Paper LR | Ours LR | Paper NN | Ours NN | Fair: period model (30 splits) | Fair: pooled model | Majority baseline | Hit rate |
|---|---|---|---|---|---|---|---|---|---|
| 1990-2018 | 3221 | 75.8 | 78.0 | 76.4 | 76.4 |  |  |  |  |
| 1990-1994 | 376 | 76.0 | 78.7 | 78.7 | 74.7 | 74.2 +- 3.9 | 59.3 | 58.5 | 59 |
| 1995-1999 | 445 | 76.4 | 79.8 | 79.7 | 80.9 | 71.7 +- 4.7 | 70.1 | 51.6 | 52 |
| 2000-2004 | 717 | 73.0 | 69.9 | 73.4 | 72.0 | 74.3 +- 3.1 | 74.0 | 62.5 | 38 |
| 2005-2009 | 1212 | 80.9 | 84.3 | 80.1 | 84.3 | 79.4 +- 2.1 | 77.7 | 64.8 | 35 |
| 2010-2014 | 313 | 82.2 | 79.4 | 83.8 | 83.9 | 89.7 +- 3.0 | 87.3 | 51.1 | 49 |
| 2015-2018 | 158 | 74.2 | 93.8 | 87.1 | 96.8 | 82.3 +- 3.9 | 85.1 | 74.2 | 26 |

### Random vs artist-grouped split (20 seeds)

| Model | Random split | Artist-grouped split | Inflation (points) | Temporal split |
|---|---|---|---|---|
| Logistic regression | 74.6 +- 1.1 | 74.8 +- 0.8 | -0.3 | 76.6 |
| k-NN (k=15) | 74.9 +- 0.7 | 74.9 +- 1.1 | +0.0 | 73.8 |
| Random forest | 76.0 +- 1.0 | 76.2 +- 1.3 | -0.2 | 79.5 |
| Gradient boosting | 75.8 +- 1.3 | 75.3 +- 1.0 | +0.6 | 76.9 |

Temporal split: 1484 training songs (<=2006), 1227 test songs (>=2007); hit rate 43% vs 37%, majority baseline on test 62.9%.

### Model comparison (10 paper features, tuned by grouped CV)

| Model | Grouped CV acc | Test acc [95% CI] | Test precision | Test recall | Test ROC AUC | Chosen hyperparameters |
|---|---|---|---|---|---|---|
| Gradient boosting | 76.8 +- 1.1 | 75.9 [72.7, 78.9] | 71.1 | 66.1 | 0.836 | l2_regularization=0.0, learning_rate=0.1, max_depth=2, max_iter=300, max_leaf_nodes=7 |
| Random forest | 76.6 +- 1.3 | 75.0 [71.9, 78.1] | 71.2 | 62.1 | 0.837 | max_depth=None, max_features=sqrt, min_samples_leaf=20 |
| Neural network (6 sigmoid units) | 75.7 +- 0.6 | 74.8 [71.8, 77.8] | 71.1 | 61.3 | 0.824 | alpha=0.1 |
| k-nearest neighbours | 75.6 +- 0.8 | 75.2 [72.0, 78.4] | 69.5 | 66.8 | 0.832 | n_neighbors=31, weights=distance |
| Logistic regression | 75.5 +- 0.7 | 74.9 [72.0, 77.9] | 72.3 | 59.5 | 0.824 | C=100 |
| RBF SVM | 75.3 +- 0.7 | 74.5 [71.3, 77.8] | 68.1 | 67.1 | 0.821 | C=10, gamma=scale |
| Neural network (32-16 ReLU) | 75.2 +- 1.0 | 74.4 [71.5, 77.4] | 69.6 | 62.8 | 0.831 | alpha=3 |
| GDA (shared covariance) | 74.5 +- 0.7 | 74.1 [70.8, 77.4] | 76.7 | 49.7 | 0.821 | - |
| Decision tree | 73.9 +- 1.4 | 75.1 [72.0, 78.1] | 71.4 | 62.1 | 0.805 | max_depth=None, min_samples_leaf=60 |
| Artist score rule | 73.3 +- 0.3 | 72.1 [68.7, 75.5] | 77.3 | 42.0 | 0.669 | - |
| Linear SVM | 73.3 +- 0.3 | 72.1 [68.7, 75.5] | 77.3 | 42.0 | 0.810 | C=0.01 |
| Naive Bayes | 69.4 +- 1.1 | 68.2 [64.6, 71.8] | 56.7 | 83.7 | 0.804 | - |
| Majority class | 60.3 +- 0.1 | 60.4 [54.3, 65.7] | 0.0 | 0.0 | 0.500 | - |

Best by CV: **Gradient boosting**.

### Improvements (cumulative, same test artists)

| Step | Grouped CV acc | Test acc [95% CI] | Test ROC AUC | vs B0, points [95% CI] |
|---|---|---|---|---|
| B0 Paper baseline (LR, 7 features, dated songs) | 73.9 +- 2.3 | 73.1 [70.1, 76.2] | 0.815 | +0.0 [+0.0, +0.0] |
| I1 + all labelled songs | 75.5 +- 1.1 | 73.1 [70.2, 76.2] | 0.814 | +0.0 [-0.7, +0.7] |
| I2 + engineered features (tuned LR) | 75.6 +- 0.7 | 74.3 [71.2, 77.5] | 0.821 | +1.2 [-0.6, +3.0] |
| I3 + nonlinear model (gradient boosting) | 76.7 +- 0.6 | 75.3 [72.2, 78.5] | 0.837 | +2.2 [-0.3, +4.6] |

I3 candidates (grouped CV): gradient boosting 76.7%, soft-voting ensemble 76.5%. Deployed model: **I3 + nonlinear model (gradient boosting)**.

I4 threshold 0.64: test precision 77.5% and recall 46.7%, vs 69.9% and 66.1% at 0.5.
