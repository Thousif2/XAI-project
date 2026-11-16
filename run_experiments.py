#!/usr/bin/env python3
"""
DS5622 Explainable AI Course Project (Track 1)
Do Irrelevant Features Mislead SHAP and LIME? An Experimental Investigation
Authors: Thousif (142301032) & Prasanna (142301027)

Everything is implemented from scratch with NumPy (no shap / lime / scikit-learn needed):
  1. A synthetic binary-classification generator with a known set of 5 informative
     features and k_irr irrelevant (optionally correlated) features.
  2. A Random Forest (CART / Gini, bootstrap, floor(sqrt(d)) features per split).
  3. Exact interventional TreeSHAP and a tabular LIME re-implementation (quartile
     discretisation, exponential kernel, weighted Ridge surrogate).
     The Random Forest's own MDI importance is the global baseline.
  4. Four experiments, each over 5 random seeds:
       Exp 1  k_irr in {0, 5, 10, 20, 40, 80}                 (RQ1, RQ2, RQ3)
       Exp 2  noise scale sigma in {0.5, 1.0, 2.0, 5.0}, k_irr = 20
       Exp 3  correlation r of distractors in {0, .3, .6, .85, .95}, k_irr = 20
       Exp 4  LIME N_samples in {100, 250, 500, 1000, 2500}, k_irr = 40
  5. CSV tables and four figures, written to ./results/

Usage:
  python3 run_experiments.py                 # full suite, about 10 min on one CPU core
  python3 run_experiments.py --figures-only  # redraw the figures from the saved CSVs
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# All CSVs and figures are written here
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Synthetic Dataset Generation with Ground-Truth Importance
# ---------------------------------------------------------------------------
# Ground-truth latent log-odds function:
# g(x) = w_0*x_0 + w_1*x_1 + w_2*(x_2^2 - 1)/sqrt(2) + w_3*sin(x_3) + w_4*x_4
# where x_0..x_4 ~ N(0, 1) are the 5 genuinely informative features with strictly
# decreasing theoretical marginal scale:
# w = [2.20, 1.65, 1.25, 0.90, 0.55] -> strict true rank order: x_0 > x_1 > x_2 > x_3 > x_4
TRUE_WEIGHTS = np.array([2.20, 1.65, 1.25, 0.90, 0.55], dtype=float)
K_INF = len(TRUE_WEIGHTS)

def generate_dataset(n_samples=600, k_irr=20, noise_std=1.0, collinearity_rho=0.0, seed=42):
    """
    Generates synthetic binary classification data with K_INF=5 informative features
    and k_irr irrelevant features.
    If collinearity_rho > 0, the first min(k_irr, K_INF) irrelevant features are correlated
    with the informative features: z_j = rho * x_j + sqrt(1 - rho^2) * eps_j.
    """
    rng = np.random.default_rng(seed)
    X_inf = rng.normal(0.0, 1.0, size=(n_samples, K_INF))
    
    # Nonlinear + linear ground-truth signal
    f0 = X_inf[:, 0]
    f1 = X_inf[:, 1]
    f2 = (X_inf[:, 2] ** 2 - 1.0) / np.sqrt(2.0)
    f3 = np.sin(X_inf[:, 3]) * 1.15
    f4 = X_inf[:, 4]
    F_comp = np.column_stack([f0, f1, f2, f3, f4])
    logits = F_comp @ TRUE_WEIGHTS + rng.normal(0.0, 0.35, size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)
    
    if k_irr > 0:
        X_irr = rng.normal(0.0, noise_std, size=(n_samples, k_irr))
        if collinearity_rho > 0.0:
            n_corr = min(k_irr, K_INF)
            for j in range(n_corr):
                eps = rng.normal(0.0, 1.0, size=n_samples)
                X_irr[:, j] = noise_std * (collinearity_rho * X_inf[:, j] + np.sqrt(1.0 - collinearity_rho**2) * eps)
        X = np.hstack([X_inf, X_irr])
    else:
        X = X_inf.copy()
        
    return X, y, probs


# ---------------------------------------------------------------------------
# 2. Fast Vectorized Random Forest Classifier & Exact TreeSHAP
# ---------------------------------------------------------------------------
class CARTNode:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'value', 'n_samples', 'impurity_gain')
    def __init__(self, feature=-1, threshold=0.0, left=None, right=None, value=0.0, n_samples=0, impurity_gain=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity_gain = impurity_gain


class FastRandomForestClassifier:
    """
    Standard Breiman (2001) Random Forest Classifier using CART Gini impurity splits,
    bootstrap aggregation, and sqrt(d) random feature subspacing at each split node.
    Stores full tree structures to support exact path-dependent & interventional TreeSHAP
    and Mean Decrease in Impurity (MDI).
    """
    def __init__(self, n_estimators=40, max_depth=6, min_samples_leaf=8, max_features='sqrt', seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.n_features_ = 0
        self.feature_importances_ = None

    def _build_tree(self, X, y, depth, rng):
        n_samples, n_features = X.shape
        pos_prob = float(np.mean(y))
        
        if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf or pos_prob == 0.0 or pos_prob == 1.0:
            return CARTNode(value=pos_prob, n_samples=n_samples)

        if self.max_features == 'sqrt':
            mtry = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            mtry = min(n_features, self.max_features)
        else:
            mtry = n_features

        feat_indices = rng.choice(n_features, size=mtry, replace=False)
        parent_gini = 2.0 * pos_prob * (1.0 - pos_prob)
        
        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        best_left_mask = None

        # Evaluate candidate quantile thresholds per selected feature
        quantiles = np.linspace(0.15, 0.85, 7)
        for f in feat_indices:
            col = X[:, f]
            thresholds = np.quantile(col, quantiles)
            for thresh in np.unique(thresholds):
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                p_left = float(np.mean(y[left_mask]))
                p_right = float(np.mean(y[~left_mask]))
                gini_left = 2.0 * p_left * (1.0 - p_left)
                gini_right = 2.0 * p_right * (1.0 - p_right)
                gain = parent_gini - (n_left / n_samples) * gini_left - (n_right / n_samples) * gini_right
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thresh)
                    best_left_mask = left_mask

        if best_feat == -1:
            return CARTNode(value=pos_prob, n_samples=n_samples)

        left_child = self._build_tree(X[best_left_mask], y[best_left_mask], depth + 1, rng)
        right_child = self._build_tree(X[~best_left_mask], y[~best_left_mask], depth + 1, rng)
        return CARTNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=pos_prob,
            n_samples=n_samples,
            impurity_gain=best_gain * n_samples
        )

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.trees = []
        mdi = np.zeros(n_features, dtype=float)
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = self._build_tree(X[boot_idx], y[boot_idx], 0, rng)
            self.trees.append(tree)
            self._accumulate_mdi(tree, mdi)

        total_mdi = np.sum(mdi)
        self.feature_importances_ = mdi / total_mdi if total_mdi > 0 else mdi
        return self

    def _accumulate_mdi(self, node, mdi):
        if node is None or node.feature == -1:
            return
        mdi[node.feature] += node.impurity_gain
        self._accumulate_mdi(node.left, mdi)
        self._accumulate_mdi(node.right, mdi)

    def _predict_tree(self, node, X):
        if node.feature == -1:
            return np.full(X.shape[0], node.value, dtype=float)
        mask = X[:, node.feature] <= node.threshold
        out = np.empty(X.shape[0], dtype=float)
        if np.any(mask):
            out[mask] = self._predict_tree(node.left, X[mask])
        if np.any(~mask):
            out[~mask] = self._predict_tree(node.right, X[~mask])
        return out

    def predict_proba(self, X):
        preds = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            preds += self._predict_tree(tree, X)
        preds /= len(self.trees)
        return np.column_stack([1.0 - preds, preds])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# 3. Exact Interventional TreeSHAP & LIME Implementations
# ---------------------------------------------------------------------------
def _tree_interventional_shap_single(node, x, X_bg):
    """
    Exact interventional TreeSHAP for ONE decision tree, explained instance x and
    background set X_bg (Lundberg et al., 2018; Chen et al., 2020).

    A tree prediction is a sum over leaves of  value(leaf) * 1[z reaches leaf].  For a
    leaf l and a background point b, let D be the path features whose condition fails
    for b but holds for x, and D' those that fail for x but hold for b. If some feature
    fails for both x and b, the leaf is unreachable for every hybrid point and
    contributes 0. Otherwise the leaf indicator is a simple "unanimity" game, whose
    Shapley values have a closed form:
        D' empty, |D| = m >= 1 :  +value / m                          for f in D
        both non-empty         :  +value / (m * C(m+m', m'))          for f in D
                                  -value / (m' * C(m+m', m))          for f in D'
        D empty, |D'| = m' >= 1:  -value / m'                         for f in D'
    Features that are not on the path get 0 (dummy-player property), and the sum over
    leaves, background points and trees gives exact Shapley values without enumerating
    2^d coalitions. validate_implementation.py checks this against brute force.
    """
    d = len(x)
    phi = np.zeros(d, dtype=float)
    n_bg = X_bg.shape[0]

    # Binomial coefficients (paths have at most max_depth distinct features)
    binom = np.ones((12, 12), dtype=float)
    for n in range(1, 12):
        for k in range(1, n):
            binom[n, k] = binom[n - 1, k - 1] + binom[n - 1, k]

    def traverse(curr, path_feats, x_conds, bg_conds_list):
        if curr.feature == -1:
            if len(path_feats) == 0 or curr.value == 0.0:
                return
            # Consolidate conditions by unique feature along the path
            uniq_feats = []
            feat_to_idx = {}
            for idx_p, f in enumerate(path_feats):
                if f not in feat_to_idx:
                    feat_to_idx[f] = len(uniq_feats)
                    uniq_feats.append(f)
            u = len(uniq_feats)
            x_sat = np.ones(u, dtype=bool)
            bg_sat = np.ones((n_bg, u), dtype=bool)
            for idx_p, f in enumerate(path_feats):
                uf = feat_to_idx[f]
                x_sat[uf] = x_sat[uf] and x_conds[idx_p]
                bg_sat[:, uf] &= bg_conds_list[idx_p]

            # If any feature fails for BOTH x and b, v(S) = 0 identically for that b
            # Vectorize across all n_bg background samples
            x_fail = ~x_sat  # shape (u,)
            bg_fail = ~bg_sat  # shape (n_bg, u)
            
            # Valid background samples where no feature fails for both x and b
            both_fail = bg_fail & x_fail[None, :]  # (n_bg, u)
            valid_bg = ~np.any(both_fail, axis=1)  # (n_bg,)
            if not np.any(valid_bg):
                return

            bg_fail_v = bg_fail[valid_bg]  # (n_valid, u) where x_sat is True on failing bg features
            n_valid = bg_fail_v.shape[0]
            m_arr = np.sum(bg_fail_v, axis=1)  # |D| for each valid b
            m_prime = int(np.sum(x_fail))      # |D'| (constant across valid b since both_fail is False)

            if m_prime == 0:
                # x reaches this leaf; for any valid b with m >= 1, each f in D gets +val / m
                pos_mask = m_arr > 0
                if np.any(pos_mask):
                    weights = (curr.value / m_arr[pos_mask])[:, None] * bg_fail_v[pos_mask]
                    contrib = np.sum(weights, axis=0) / n_bg
                    for uf, f in enumerate(uniq_feats):
                        phi[f] += contrib[uf]
            else:
                # m_prime >= 1
                # For b with m == 0 (b reaches leaf), each f in D' gets -val / m_prime
                m_zero_count = int(np.sum(m_arr == 0))
                if m_zero_count > 0:
                    w_neg0 = -curr.value * (m_zero_count / n_bg) / m_prime
                    for uf, f in enumerate(uniq_feats):
                        if x_fail[uf]:
                            phi[f] += w_neg0

                # For b with m >= 1:
                pos_mask = m_arr > 0
                if np.any(pos_mask):
                    m_pos = m_arr[pos_mask]
                    # Positive weight for f in D: +val / (m * binom[m + m_prime, m_prime])
                    denom_pos = np.array([m_val * binom[m_val + m_prime, m_prime] for m_val in m_pos])
                    w_pos = (curr.value / denom_pos)[:, None] * bg_fail_v[pos_mask]
                    contrib_pos = np.sum(w_pos, axis=0) / n_bg
                    for uf, f in enumerate(uniq_feats):
                        phi[f] += contrib_pos[uf]

                    # Negative weight for f in D': -val / (m_prime * binom[m + m_prime, m])
                    denom_neg = np.array([m_prime * binom[m_val + m_prime, m_val] for m_val in m_pos])
                    w_neg_sum = np.sum(-curr.value / denom_neg) / n_bg
                    for uf, f in enumerate(uniq_feats):
                        if x_fail[uf]:
                            phi[f] += w_neg_sum
            return

        f = curr.feature
        t = curr.threshold
        x_left = bool(x[f] <= t)
        bg_left = (X_bg[:, f] <= t)

        traverse(curr.left, path_feats + [f], x_conds + [x_left], bg_conds_list + [bg_left])
        traverse(curr.right, path_feats + [f], x_conds + [not x_left], bg_conds_list + [~bg_left])

    traverse(node, [], [], [])
    return phi


def compute_treeshap_attributions(model, X_eval, X_bg):
    """
    Computes exact Interventional TreeSHAP attributions for all instances in X_eval
    across all trees in the Random Forest ensemble.
    Verifies local accuracy: sum_j phi_j(x) == f(x) - E[f(X_bg)].
    """
    n_eval, d = X_eval.shape
    phis = np.zeros((n_eval, d), dtype=float)
    for i in range(n_eval):
        x = X_eval[i]
        phi_i = np.zeros(d, dtype=float)
        for tree in model.trees:
            phi_i += _tree_interventional_shap_single(tree, x, X_bg)
        phis[i] = phi_i / len(model.trees)
    return phis


class TabularLIMEExplainer:
    """
    Implements Local Interpretable Model-agnostic Explanations (LIME) for tabular data
    following Ribeiro et al. (2016) (`lime.lime_tabular.LimeTabularExplainer`):
    - Quartile discretization of continuous features into 4 quantile bins.
    - Binary interpretable representation z' in {0, 1}^d sampled uniformly via
      independent Bernoulli perturbations, with continuous values reconstructed from
      empirical bin means/stds.
    - Exponential kernel weighting pi_x(z') = exp(-D(x', z')^2 / sigma^2) where
      D(x', z') is Euclidean distance in interpretable space and sigma = 0.75 * sqrt(d).
    - Weighted Ridge regression surrogate fit locally to black-box probabilities f(z).
    """
    def __init__(self, X_train, kernel_scale=0.75):
        self.n_train, self.d = X_train.shape
        self.kernel_scale = kernel_scale
        self.quantiles = np.percentile(X_train, [25, 50, 75], axis=0)  # shape (3, d)
        self.bins = np.zeros((self.n_train, self.d), dtype=int)
        for j in range(self.d):
            self.bins[:, j] = np.digitize(X_train[:, j], self.quantiles[:, j])
        
        # Precompute bin statistics and empirical frequencies for fast inverse sampling
        self.bin_means = np.zeros((4, self.d), dtype=float)
        self.bin_stds = np.zeros((4, self.d), dtype=float)
        self.bin_probs = np.zeros((4, self.d), dtype=float)
        for j in range(self.d):
            for b in range(4):
                mask = (self.bins[:, j] == b)
                cnt = int(np.sum(mask))
                if cnt > 0:
                    self.bin_means[b, j] = float(np.mean(X_train[mask, j]))
                    self.bin_stds[b, j] = max(1e-4, float(np.std(X_train[mask, j])))
                    self.bin_probs[b, j] = cnt / self.n_train
                else:
                    self.bin_means[b, j] = float(np.mean(X_train[:, j]))
                    self.bin_stds[b, j] = max(1e-4, float(np.std(X_train[:, j])))
                    self.bin_probs[b, j] = 0.25
            self.bin_probs[:, j] /= np.sum(self.bin_probs[:, j])

    def explain_instance(self, x, predict_proba_fn, num_samples=500, ridge_alpha=1.0, seed=42):
        rng = np.random.default_rng(seed)
        d = self.d
        # Discretize instance x
        x_bins = np.array([int(np.digitize(x[j], self.quantiles[:, j])) for j in range(d)])
        
        # Sample interpretable binary matrix Z_prime in {0, 1}^{(num_samples, d)}
        # First row is always all-ones (the original instance x)
        Z_prime = rng.integers(0, 2, size=(num_samples, d))
        Z_prime[0, :] = 1
        
        # Reconstruct continuous perturbations Z_cont
        Z_cont = np.empty((num_samples, d), dtype=float)
        Z_cont[0, :] = x
        
        for j in range(d):
            # Sample alternative bins according to empirical training distribution
            sampled_bins = rng.choice(4, size=num_samples - 1, p=self.bin_probs[:, j])
            # Where Z_prime[1:, j] == 1, keep x's bin; else use sampled bin
            active_bins = np.where(Z_prime[1:, j] == 1, x_bins[j], sampled_bins)
            means = self.bin_means[active_bins, j]
            stds = self.bin_stds[active_bins, j]
            Z_cont[1:, j] = rng.normal(means, stds)

        # Query black-box model
        y_hat = predict_proba_fn(Z_cont)[:, 1]
        
        # Compute exponential kernel weights in interpretable space
        # Distance between Z_prime[0] (all ones) and Z_prime[i] is sqrt(sum(1 - Z_prime[i]))
        sq_dists = np.sum((1.0 - Z_prime) ** 2, axis=1)
        kernel_width = self.kernel_scale * np.sqrt(d)
        weights = np.exp(-sq_dists / (kernel_width ** 2 + 1e-12))
        
        # Fit weighted Ridge regression: min_{w, b} sum_i w_i (y_hat_i - b - Z_prime_i @ w)^2 + alpha ||w||^2
        W_sqrt = np.sqrt(weights)
        X_design = np.column_stack([np.ones(num_samples), Z_prime])
        X_w = X_design * W_sqrt[:, None]
        y_w = y_hat * W_sqrt
        
        reg = np.eye(d + 1) * ridge_alpha
        reg[0, 0] = 1e-6  # Do not regularize intercept
        theta = np.linalg.solve(X_w.T @ X_w + reg, X_w.T @ y_w)
        return theta[1:]  # Return feature attributions w in R^d

    def explain_batch(self, X_eval, predict_proba_fn, num_samples=500, seed=42):
        n_eval = X_eval.shape[0]
        attributions = np.zeros((n_eval, self.d), dtype=float)
        for i in range(n_eval):
            attributions[i] = self.explain_instance(
                X_eval[i], predict_proba_fn, num_samples=num_samples, seed=seed + i * 17
            )
        return attributions


# ---------------------------------------------------------------------------
# 4. Evaluation Metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true, y_pred, y_prob):
    acc = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # ROC-AUC via Wilcoxon-Mann-Whitney rank statistic
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        auc = float(np.mean(pos_probs[:, None] > neg_probs[None, :]) + 0.5 * np.mean(pos_probs[:, None] == neg_probs[None, :]))
    else:
        auc = 0.5
    return acc, f1, auc


def evaluate_explanations(attr_matrix, k_inf=K_INF):
    """
    Given local attribution matrix of shape (n_eval, d), computes:
    1. Global Top-k Feature Recovery (Precision@5 on mean |phi_j|): fraction of top-5 attributed
       features that belong to the true informative set {0, 1, 2, 3, 4}.
    2. Local Instance-Level Top-k Recovery: average Precision@5 across individual test instances.
    3. Spearman Rank Correlation (rho_inf) between estimated importance of informative features
       and their true ground-truth ordering w = [2.20, 1.65, 1.25, 0.90, 0.55].
    4. Spurious Attribution Mass Ratio (SAMR): fraction of total absolute attribution mass
       assigned to irrelevant features j >= k_inf:
       SAMR = sum_{j >= k_inf} E[|phi_j|] / sum_{j=1}^d E[|phi_j|].
    """
    abs_attr = np.abs(attr_matrix)
    global_imp = np.mean(abs_attr, axis=0)
    d = len(global_imp)
    
    # 1. Global Top-k recovery
    top_k_global = set(np.argsort(global_imp)[::-1][:k_inf])
    true_set = set(range(k_inf))
    global_recovery = len(top_k_global.intersection(true_set)) / float(k_inf)
    
    # 2. Local instance-level Top-k recovery
    local_rec_list = []
    local_spearman_list = []
    for i in range(abs_attr.shape[0]):
        top_k_inst = set(np.argsort(abs_attr[i])[::-1][:k_inf])
        local_rec_list.append(len(top_k_inst.intersection(true_set)) / float(k_inf))
        corr_i, _ = spearmanr(TRUE_WEIGHTS, abs_attr[i, :k_inf])
        if np.isnan(corr_i):
            corr_i = 0.0
        local_spearman_list.append(float(corr_i))
    local_recovery = float(np.mean(local_rec_list))
    local_spearman = float(np.mean(local_spearman_list))
    
    # 3. Global Spearman rank correlation on informative features
    global_spearman, _ = spearmanr(TRUE_WEIGHTS, global_imp[:k_inf])
    if np.isnan(global_spearman):
        global_spearman = 0.0
        
    # 4. Spurious Attribution Mass Ratio (SAMR)
    total_mass = float(np.sum(global_imp))
    if d > k_inf and total_mass > 0:
        samr = float(np.sum(global_imp[k_inf:]) / total_mass)
    else:
        samr = 0.0
        
    return {
        "global_recovery": global_recovery,
        "local_recovery": local_recovery,
        "global_spearman": float(global_spearman),
        "local_spearman": local_spearman,
        "samr": samr,
        "global_imp": global_imp
    }


def compute_explanation_stability(attr_run1, attr_run2):
    """
    Measures local explanation stability between two runs (e.g., different random seeds
    for LIME perturbations or background subsamples for SHAP) via mean pairwise Cosine
    Similarity and Top-5 Jaccard Index across instances.
    """
    n_eval = attr_run1.shape[0]
    cos_sims = []
    jaccards = []
    for i in range(n_eval):
        v1 = attr_run1[i]
        v2 = attr_run2[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-12 and norm2 > 1e-12:
            cos_sims.append(float(np.dot(v1, v2) / (norm1 * norm2)))
        else:
            cos_sims.append(0.0)
        t1 = set(np.argsort(np.abs(v1))[::-1][:K_INF])
        t2 = set(np.argsort(np.abs(v2))[::-1][:K_INF])
        jaccards.append(len(t1.intersection(t2)) / float(len(t1.union(t2))))
    return float(np.mean(cos_sims)), float(np.mean(jaccards))


# ---------------------------------------------------------------------------
# 5. Main Experimental Suite Execution
# ---------------------------------------------------------------------------
def run_all_experiments():
    seeds = [42, 101, 202, 303, 404]
    n_train = 450
    n_test = 150
    n_explain = 50  # Instances explained per seed
    n_bg = 80       # Background reference samples for SHAP

    print("=== Running Experiment 1: Progressive Irrelevant Features ===")
    k_irr_values = [0, 5, 10, 20, 40, 80]
    exp1_records = []

    for k_irr in k_irr_values:
        for seed in seeds:
            X, y, _ = generate_dataset(n_samples=n_train + n_test, k_irr=k_irr, noise_std=1.0, collinearity_rho=0.0, seed=seed)
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]
            X_eval = X_test[:n_explain]
            X_bg = X_train[:n_bg]

            rf = FastRandomForestClassifier(n_estimators=40, max_depth=6, min_samples_leaf=8, max_features='sqrt', seed=seed)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1]
            acc, f1, auc = compute_classification_metrics(y_test, y_pred, y_prob)

            # SHAP Attributions (Run 1 & Run 2 with different background subsample for stability)
            t0 = time.time()
            shap_attr1 = compute_treeshap_attributions(rf, X_eval, X_bg)
            shap_time = time.time() - t0
            shap_attr2 = compute_treeshap_attributions(rf, X_eval, X_train[40:40+n_bg])
            shap_eval = evaluate_explanations(shap_attr1)
            shap_cos, shap_jac = compute_explanation_stability(shap_attr1, shap_attr2)

            # Check local accuracy error of TreeSHAP
            fx = rf.predict_proba(X_eval)[:, 1]
            efx = float(np.mean(rf.predict_proba(X_bg)[:, 1]))
            shap_local_err = float(np.max(np.abs(np.sum(shap_attr1, axis=1) - (fx - efx))))

            # LIME Attributions (Run 1 & Run 2 with different perturbation seeds for stability)
            lime_exp = TabularLIMEExplainer(X_train, kernel_scale=0.75)
            t0 = time.time()
            lime_attr1 = lime_exp.explain_batch(X_eval, rf.predict_proba, num_samples=500, seed=seed + 1000)
            lime_time = time.time() - t0
            lime_attr2 = lime_exp.explain_batch(X_eval, rf.predict_proba, num_samples=500, seed=seed + 5000)
            lime_eval = evaluate_explanations(lime_attr1)
            lime_cos, lime_jac = compute_explanation_stability(lime_attr1, lime_attr2)

            # Baseline: Random Forest MDI & Permutation Importance
            mdi_eval = evaluate_explanations(np.tile(rf.feature_importances_, (n_explain, 1)))

            exp1_records.append({
                "k_irr": k_irr,
                "seed": seed,
                "acc": acc,
                "f1": f1,
                "auc": auc,
                "shap_global_rec": shap_eval["global_recovery"],
                "shap_local_rec": shap_eval["local_recovery"],
                "shap_global_rho": shap_eval["global_spearman"],
                "shap_local_rho": shap_eval["local_spearman"],
                "shap_samr": shap_eval["samr"],
                "shap_cos_stab": shap_cos,
                "shap_jac_stab": shap_jac,
                "shap_time": shap_time,
                "shap_local_err": shap_local_err,
                "lime_global_rec": lime_eval["global_recovery"],
                "lime_local_rec": lime_eval["local_recovery"],
                "lime_global_rho": lime_eval["global_spearman"],
                "lime_local_rho": lime_eval["local_spearman"],
                "lime_samr": lime_eval["samr"],
                "lime_cos_stab": lime_cos,
                "lime_jac_stab": lime_jac,
                "lime_time": lime_time,
                "mdi_global_rec": mdi_eval["global_recovery"],
                "mdi_samr": mdi_eval["samr"],
            })

    df_exp1 = pd.DataFrame(exp1_records)
    df_exp1.to_csv(os.path.join(RESULTS_DIR, "exp1_progressive_irrelevant.csv"), index=False)

    print("=== Running Experiment 2: Noise Amplitude Sensitivity (k_irr = 20) ===")
    noise_levels = [0.5, 1.0, 2.0, 5.0]
    exp2_records = []
    for sigma in noise_levels:
        for seed in seeds:
            X, y, _ = generate_dataset(n_samples=n_train + n_test, k_irr=20, noise_std=sigma, collinearity_rho=0.0, seed=seed)
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]
            X_eval = X_test[:n_explain]
            X_bg = X_train[:n_bg]

            rf = FastRandomForestClassifier(n_estimators=40, max_depth=6, min_samples_leaf=8, max_features='sqrt', seed=seed)
            rf.fit(X_train, y_train)
            shap_attr = compute_treeshap_attributions(rf, X_eval, X_bg)
            lime_exp = TabularLIMEExplainer(X_train, kernel_scale=0.75)
            lime_attr = lime_exp.explain_batch(X_eval, rf.predict_proba, num_samples=500, seed=seed + 2000)

            s_ev = evaluate_explanations(shap_attr)
            l_ev = evaluate_explanations(lime_attr)
            exp2_records.append({
                "noise_std": sigma,
                "seed": seed,
                "shap_local_rec": s_ev["local_recovery"],
                "shap_samr": s_ev["samr"],
                "shap_local_rho": s_ev["local_spearman"],
                "lime_local_rec": l_ev["local_recovery"],
                "lime_samr": l_ev["samr"],
                "lime_local_rho": l_ev["local_spearman"],
            })
    df_exp2 = pd.DataFrame(exp2_records)
    df_exp2.to_csv(os.path.join(RESULTS_DIR, "exp2_noise_sensitivity.csv"), index=False)

    print("=== Running Experiment 3: Collinear Distractor Sensitivity (k_irr = 20) ===")
    rho_levels = [0.0, 0.3, 0.6, 0.85, 0.95]
    exp3_records = []
    for rho in rho_levels:
        for seed in seeds:
            X, y, _ = generate_dataset(n_samples=n_train + n_test, k_irr=20, noise_std=1.0, collinearity_rho=rho, seed=seed)
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]
            X_eval = X_test[:n_explain]
            X_bg = X_train[:n_bg]

            rf = FastRandomForestClassifier(n_estimators=40, max_depth=6, min_samples_leaf=8, max_features='sqrt', seed=seed)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1]
            acc, f1, auc = compute_classification_metrics(y_test, y_pred, y_prob)

            shap_attr = compute_treeshap_attributions(rf, X_eval, X_bg)
            lime_exp = TabularLIMEExplainer(X_train, kernel_scale=0.75)
            lime_attr = lime_exp.explain_batch(X_eval, rf.predict_proba, num_samples=500, seed=seed + 3000)

            s_ev = evaluate_explanations(shap_attr)
            l_ev = evaluate_explanations(lime_attr)
            exp3_records.append({
                "collinearity_rho": rho,
                "seed": seed,
                "acc": acc,
                "f1": f1,
                "shap_global_rec": s_ev["global_recovery"],
                "shap_local_rec": s_ev["local_recovery"],
                "shap_samr": s_ev["samr"],
                "shap_local_rho": s_ev["local_spearman"],
                "lime_global_rec": l_ev["global_recovery"],
                "lime_local_rec": l_ev["local_recovery"],
                "lime_samr": l_ev["samr"],
                "lime_local_rho": l_ev["local_spearman"],
            })
    df_exp3 = pd.DataFrame(exp3_records)
    df_exp3.to_csv(os.path.join(RESULTS_DIR, "exp3_collinearity_sensitivity.csv"), index=False)

    print("=== Running Experiment 4: LIME Sample Size Ablation (k_irr = 40) ===")
    sample_sizes = [100, 250, 500, 1000, 2500]
    exp4_records = []
    for n_samp in sample_sizes:
        for seed in seeds:
            X, y, _ = generate_dataset(n_samples=n_train + n_test, k_irr=40, noise_std=1.0, collinearity_rho=0.0, seed=seed)
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]
            X_eval = X_test[:35]

            rf = FastRandomForestClassifier(n_estimators=40, max_depth=6, min_samples_leaf=8, max_features='sqrt', seed=seed)
            rf.fit(X_train, y_train)
            lime_exp = TabularLIMEExplainer(X_train, kernel_scale=0.75)
            t0 = time.time()
            l1 = lime_exp.explain_batch(X_eval, rf.predict_proba, num_samples=n_samp, seed=seed + 4000)
            elapsed = time.time() - t0
            l2 = lime_exp.explain_batch(X_eval, rf.predict_proba, num_samples=n_samp, seed=seed + 8000)
            l_ev = evaluate_explanations(l1)
            cos_s, jac_s = compute_explanation_stability(l1, l2)
            exp4_records.append({
                "num_samples": n_samp,
                "seed": seed,
                "lime_local_rec": l_ev["local_recovery"],
                "lime_local_rho": l_ev["local_spearman"],
                "lime_samr": l_ev["samr"],
                "lime_cos_stab": cos_s,
                "lime_jac_stab": jac_s,
                "lime_time": elapsed
            })
    df_exp4 = pd.DataFrame(exp4_records)
    df_exp4.to_csv(os.path.join(RESULTS_DIR, "exp4_lime_ablation.csv"), index=False)

    return df_exp1, df_exp2, df_exp3, df_exp4


# ---------------------------------------------------------------------------
# 6. Figures (drawn from the CSV files written by run_all_experiments)
# ---------------------------------------------------------------------------
C_SHAP, C_LIME, C_MDI, C_PERF = '#1e3a8a', '#d97706', '#64748b', '#0d9488'


def _style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8.5, 'axes.titlesize': 9.5,
        'axes.titleweight': 'bold', 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 7, 'axes.grid': True,
        'grid.linestyle': '--', 'grid.alpha': 0.35,
    })


def _save(fig, name):
    fig.savefig(os.path.join(RESULTS_DIR, name), bbox_inches='tight', dpi=300)
    plt.close(fig)


def generate_figures(df1, df2, df3, df4):
    _style()
    XL_K = 'Number of irrelevant features ($k_{\\mathrm{irr}}$)'

    # ---- Figure 1: top-5 recovery and attribution mass on noise vs k_irr ----------------
    g = df1.groupby("k_irr").agg(["mean", "std"])
    k = g.index.values
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)

    a = ax[0]
    a.errorbar(k, g["shap_local_rec"]["mean"] * 100, g["shap_local_rec"]["std"] * 100, fmt='-o',
               color=C_SHAP, lw=1.8, capsize=3, ms=4.5, label='TreeSHAP, local (per instance)')
    a.errorbar(k, g["lime_local_rec"]["mean"] * 100, g["lime_local_rec"]["std"] * 100, fmt='--s',
               color=C_LIME, lw=1.8, capsize=3, ms=4.5, label='LIME, local (per instance)')
    a.plot(k, g["shap_global_rec"]["mean"] * 100, '-.', color=C_SHAP, alpha=.55, lw=1.2,
           marker='o', mfc='white', ms=4, label='TreeSHAP, global (mean |$\\phi$|)')
    a.plot(k, g["lime_global_rec"]["mean"] * 100, '-.', color=C_LIME, alpha=.65, lw=1.2,
           marker='s', mfc='white', ms=4, label='LIME, global (mean |$w$|)')
    a.plot(k, g["mdi_global_rec"]["mean"] * 100, ':^', color=C_MDI, lw=1.4, ms=4, label='RF MDI, global')
    a.set_xlabel(XL_K); a.set_ylabel('Top-5 recovery (%)')
    a.set_title('(a) Informative-feature recovery'); a.set_ylim(15, 106)
    a.legend(loc='lower left', frameon=True, framealpha=.9)

    b = ax[1]
    b.errorbar(k, g["shap_samr"]["mean"] * 100, g["shap_samr"]["std"] * 100, fmt='-o', color=C_SHAP,
               lw=1.8, capsize=3, ms=4.5, label='TreeSHAP')
    b.errorbar(k, g["lime_samr"]["mean"] * 100, g["lime_samr"]["std"] * 100, fmt='--s', color=C_LIME,
               lw=1.8, capsize=3, ms=4.5, label='LIME')
    b.plot(k, g["mdi_samr"]["mean"] * 100, ':^', color=C_MDI, lw=1.4, ms=4, label='RF MDI')
    b.plot(k, k / (k + K_INF) * 100, '-', color='#9ca3af', lw=1.0, label='Uniform spread ($k_{irr}/d$)')
    b.set_xlabel(XL_K); b.set_ylabel('Attribution mass on noise (%)')
    b.set_title('(b) Share of attribution given to noise'); b.set_ylim(-2, 100)
    b.legend(loc='lower right', frameon=True, framealpha=.9)
    _save(fig, "fig1_feature_recovery_samr.png")

    # ---- Figure 2: ranking fidelity, and accuracy vs stability --------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    a = ax[0]
    a.errorbar(k, g["shap_local_rho"]["mean"], g["shap_local_rho"]["std"], fmt='-o', color=C_SHAP,
               lw=1.8, capsize=3, ms=4.5, label='TreeSHAP, local')
    a.errorbar(k, g["lime_local_rho"]["mean"], g["lime_local_rho"]["std"], fmt='--s', color=C_LIME,
               lw=1.8, capsize=3, ms=4.5, label='LIME, local')
    a.plot(k, g["shap_global_rho"]["mean"], '-.', color=C_SHAP, alpha=.55, lw=1.2, marker='o',
           mfc='white', ms=4, label='TreeSHAP, global')
    a.plot(k, g["lime_global_rho"]["mean"], '-.', color=C_LIME, alpha=.65, lw=1.2, marker='s',
           mfc='white', ms=4, label='LIME, global')
    a.set_xlabel(XL_K); a.set_ylabel('Spearman $\\rho$ (5 informative features)')
    a.set_title('(a) Ranking among informative features'); a.set_ylim(0.1, 1.05)
    a.legend(loc='lower left', ncol=2, frameon=True, framealpha=.9)

    b = ax[1]
    b.plot(k, g["auc"]["mean"] * 100, '-^', color=C_PERF, lw=1.8, ms=4.5, label='Model ROC-AUC')
    b.plot(k, g["f1"]["mean"] * 100, '--v', color='#059669', lw=1.5, ms=4.5, label='Model F1')
    b.plot(k, g["shap_cos_stab"]["mean"] * 100, '-o', color=C_SHAP, lw=1.8, ms=4.5, label='TreeSHAP stability (cosine)')
    b.plot(k, g["lime_cos_stab"]["mean"] * 100, '--s', color=C_LIME, lw=1.8, ms=4.5, label='LIME stability (cosine)')
    b.set_xlabel(XL_K); b.set_ylabel('Score (%)')
    b.set_title('(b) Model performance vs. explanation stability'); b.set_ylim(25, 103)
    b.legend(loc='lower left', frameon=True, framealpha=.9)
    _save(fig, "fig2_ranking_and_model_perf.png")

    # ---- Figure 3: correlated distractors and noise scale --------------------------------
    g2 = df2.groupby("noise_std").agg(["mean", "std"])
    g3 = df3.groupby("collinearity_rho").agg(["mean", "std"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    for a, gg, xs, xl, tt in [
            (ax[0], g3, g3.index.values, 'Correlation $r$ between distractor and informative feature',
             '(a) Correlated distractors ($k_{irr}=20$)'),
            (ax[1], g2, g2.index.values, 'Noise standard deviation $\\sigma_{\\mathrm{noise}}$',
             '(b) Noise scale ($k_{irr}=20$)')]:
        h1 = a.errorbar(xs, gg["shap_local_rec"]["mean"] * 100, gg["shap_local_rec"]["std"] * 100, fmt='-o',
                        color=C_SHAP, lw=1.8, capsize=3, ms=4.5, label='TreeSHAP top-5 recovery')
        h2 = a.errorbar(xs, gg["lime_local_rec"]["mean"] * 100, gg["lime_local_rec"]["std"] * 100, fmt='--s',
                        color=C_LIME, lw=1.8, capsize=3, ms=4.5, label='LIME top-5 recovery')
        h3, = a.plot(xs, gg["shap_samr"]["mean"] * 100, '-.d', color='#60a5fa', lw=1.4, ms=4, label='TreeSHAP noise mass')
        h4, = a.plot(xs, gg["lime_samr"]["mean"] * 100, ':^', color='#f59e0b', lw=1.4, ms=4, label='LIME noise mass')
        a.set_xlabel(xl); a.set_ylabel('Percentage (%)'); a.set_title(tt); a.set_ylim(10, 100)
    ax[1].set_xticks([0.5, 1, 2, 5])
    fig.legend([h1, h2, h3, h4], [h.get_label() for h in (h1, h2, h3, h4)], loc='outside lower center', ncol=4,
               frameon=False)
    _save(fig, "fig3_sensitivity_collinearity_noise.png")

    # ---- Figure 4: LIME sample-size ablation ---------------------------------------------
    g4 = df4.groupby("num_samples").agg(["mean", "std"])
    n = g4.index.values
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    a = ax[0]
    a.errorbar(n, g4["lime_local_rec"]["mean"] * 100, g4["lime_local_rec"]["std"] * 100, fmt='-s', color=C_LIME,
               lw=1.8, capsize=3, ms=4.5, label='LIME top-5 recovery')
    a.errorbar(n, g4["lime_cos_stab"]["mean"] * 100, g4["lime_cos_stab"]["std"] * 100, fmt='--o', color=C_SHAP,
               lw=1.8, capsize=3, ms=4.5, label='LIME stability (cosine)')
    a.errorbar(n, g4["lime_jac_stab"]["mean"] * 100, g4["lime_jac_stab"]["std"] * 100, fmt=':^', color=C_MDI,
               lw=1.6, capsize=3, ms=4.5, label='LIME stability (top-5 Jaccard)')
    a.set_xscale('log'); a.set_xticks(n); a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.minorticks_off()
    a.set_xlabel('LIME perturbation samples ($N_{\\mathrm{samples}}$)'); a.set_ylabel('Percentage (%)')
    a.set_title('(a) Recovery and stability'); a.set_ylim(10, 105)
    a.legend(loc='lower right', frameon=True, framealpha=.9)

    b = ax[1]
    b.errorbar(n, g4["lime_samr"]["mean"] * 100, g4["lime_samr"]["std"] * 100, fmt='-d', color='#dc2626', lw=1.8,
               capsize=3, ms=4.5, label='LIME noise mass (left)')
    b2 = b.twinx(); b2.grid(False)
    b2.plot(n, g4["lime_time"]["mean"], ':^', color='#475569', lw=1.6, ms=4.5, label='Runtime for 35 instances (right)')
    b2.set_ylim(0, 2.4); b2.set_ylabel('Seconds', color='#475569')
    b.set_xscale('log'); b.set_xticks(n); b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    b.minorticks_off()
    b.set_xlabel('LIME perturbation samples ($N_{\\mathrm{samples}}$)'); b.set_ylabel('Attribution mass on noise (%)', color='#dc2626')
    b.set_title('(b) Noise mass and cost'); b.set_ylim(15, 85)
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, loc='lower center', frameon=True, framealpha=.9)
    _save(fig, "fig4_lime_ablation_stability.png")


def load_results():
    names = ["exp1_progressive_irrelevant", "exp2_noise_sensitivity",
             "exp3_collinearity_sensitivity", "exp4_lime_ablation"]
    return [pd.read_csv(os.path.join(RESULTS_DIR, n + ".csv")) for n in names]


def main():
    parser = argparse.ArgumentParser(description="DS5622 XAI project: full experiment suite")
    parser.add_argument("--figures-only", action="store_true",
                        help="skip the experiments and redraw the figures from results/*.csv")
    args = parser.parse_args()

    t_start = time.time()
    if args.figures_only:
        df1, df2, df3, df4 = load_results()
    else:
        df1, df2, df3, df4 = run_all_experiments()
    generate_figures(df1, df2, df3, df4)

    # Short summary tables (means over the 5 seeds)
    print("\n--- Experiment 1: mean over 5 seeds ---")
    cols1 = ["acc", "f1", "auc", "shap_global_rec", "shap_local_rec", "shap_local_rho", "shap_samr",
             "shap_cos_stab", "shap_jac_stab", "shap_local_err", "lime_global_rec", "lime_local_rec",
             "lime_local_rho", "lime_samr", "lime_cos_stab", "lime_jac_stab", "mdi_global_rec", "mdi_samr"]
    print(df1.groupby("k_irr")[cols1].mean().round(4).to_string())
    print("\n--- Experiment 2: noise std ---")
    print(df2.drop(columns="seed").groupby("noise_std").mean().round(4).to_string())
    print("\n--- Experiment 3: correlation of distractors ---")
    print(df3.drop(columns="seed").groupby("collinearity_rho").mean().round(4).to_string())
    print("\n--- Experiment 4: LIME N_samples (runtime is machine dependent) ---")
    print(df4.drop(columns="seed").groupby("num_samples").mean().round(4).to_string())
    print(f"\nDone: CSVs and 4 figures are in ./{RESULTS_DIR}/  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
