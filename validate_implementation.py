#!/usr/bin/env python3
"""
Sanity checks for our from-scratch implementations (run: python3 validate_implementation.py).
  1. TreeSHAP vs brute-force Shapley values (interventional, same background).
  2. Local accuracy of TreeSHAP over all runs of Experiment 1 (needs results/ CSVs).
  3. How often the forest splits on noise features, and how often a node has no informative
     candidate among its floor(sqrt(d)) random features.
  4. Our Random Forest vs scikit-learn's (only if scikit-learn is installed).
"""
import itertools, math
import numpy as np, pandas as pd
import run_experiments as R


def brute_force_shapley(model, x, X_bg):
    d = len(x); cache = {}
    def v(S):
        if S not in cache:
            Z = X_bg.copy()
            if S: Z[:, list(S)] = x[list(S)]
            cache[S] = model.predict_proba(Z)[:, 1].mean()
        return cache[S]
    phi = np.zeros(d)
    for j in range(d):
        others = [i for i in range(d) if i != j]
        for r in range(d):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
                phi[j] += w * (v(tuple(sorted(S + (j,)))) - v(S))
    return phi


def split_features(node, out):
    if node is None or node.feature == -1: return
    out.append(node.feature); split_features(node.left, out); split_features(node.right, out)


print("1. TreeSHAP vs brute-force Shapley (3 test points, seed 42)")
for k_irr in [0, 5]:
    X, y, _ = R.generate_dataset(600, k_irr, 1.0, 0.0, 42)
    rf = R.FastRandomForestClassifier(40, 6, 8, 'sqrt', 42).fit(X[:450], y[:450])
    bg, test = X[:80], X[450:453]
    err = max(np.abs(R.compute_treeshap_attributions(rf, test[i:i+1], bg)[0] - brute_force_shapley(rf, test[i], bg)).max()
              for i in range(3))
    print(f"   k_irr={k_irr:2d}: max |difference| = {err:.1e}")

try:
    d1 = pd.read_csv("results/exp1_progressive_irrelevant.csv")
    print(f"2. Largest local-accuracy error over all {len(d1)} Experiment-1 runs: {d1.shap_local_err.max():.1e}")
except FileNotFoundError:
    print("2. skipped (run run_experiments.py first)")

print("3. Splits on noise features (mean over 5 seeds)")
print("   k_irr   d  mtry  P(no informative candidate)  noise splits  uniform share")
for k_irr in [5, 10, 20, 40, 80]:
    d = 5 + k_irr; mtry = int(np.sqrt(d))
    p = np.prod([(k_irr - i) / (d - i) for i in range(mtry)])
    fr = []
    for s in [42, 101, 202, 303, 404]:
        X, y, _ = R.generate_dataset(600, k_irr, 1.0, 0.0, s)
        rf = R.FastRandomForestClassifier(40, 6, 8, 'sqrt', s).fit(X[:450], y[:450])
        feats = []
        for t in rf.trees: split_features(t, feats)
        fr.append(np.mean(np.array(feats) >= 5))
    print(f"   {k_irr:5d} {d:3d} {mtry:5d}  {p*100:12.1f}%  {np.mean(fr)*100:14.1f}%  {k_irr/d*100:10.1f}%")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    print("4. ROC-AUC (%), our Random Forest vs scikit-learn (same hyper-parameters, 5 seeds)")
    for k_irr in [0, 20, 80]:
        a, b = [], []
        for s in [42, 101, 202, 303, 404]:
            X, y, _ = R.generate_dataset(600, k_irr, 1.0, 0.0, s)
            rf = R.FastRandomForestClassifier(40, 6, 8, 'sqrt', s).fit(X[:450], y[:450])
            a.append(R.compute_classification_metrics(y[450:], rf.predict(X[450:]), rf.predict_proba(X[450:])[:, 1])[2])
            sk = RandomForestClassifier(40, max_depth=6, min_samples_leaf=8, random_state=s).fit(X[:450], y[:450])
            b.append(roc_auc_score(y[450:], sk.predict_proba(X[450:])[:, 1]))
        print(f"   k_irr={k_irr:2d}: ours {np.mean(a)*100:.2f}   sklearn {np.mean(b)*100:.2f}")
except ImportError:
    print("4. skipped (scikit-learn not installed)")
