#!/usr/bin/env python3
"""
Cross-check of our from-scratch explainers against the standard `shap` and `lime` packages
(the packages named in our proposal).   Needs:  pip install shap lime scikit-learn

  A. Exact check: shap's Exact explainer with an Independent (= interventional) masker is run on
     OUR forest with the same 80 background points; we report max |phi_shap - phi_ours| for
     d = 5 and d = 10 (k_irr = 0, 5).
  B. SHAP end-to-end: shap.TreeExplainer (interventional, probability output) on a scikit-learn
     Random Forest with the same hyper-parameters, versus our TreeSHAP on our forest.
  C. LIME end-to-end: lime.lime_tabular on OUR forest (same model, same 50 points, same 500
     samples) versus our LIME re-implementation: recovery, noise mass and stability.

Usage:
  python3 compare_with_libraries.py                       # default: 3 seeds, k_irr in {0, 20, 80}
  python3 compare_with_libraries.py --full                # 5 seeds, all six k_irr values (slow)
  python3 compare_with_libraries.py --seeds 42 --k 20 --instances 10   # quick smoke test
Writes results/library_comparison.csv
"""
import argparse, os, time
import numpy as np, pandas as pd
import run_experiments as R

p = argparse.ArgumentParser()
p.add_argument("--full", action="store_true")
p.add_argument("--seeds", type=int, nargs="+")
p.add_argument("--k", type=int, nargs="+")
p.add_argument("--instances", type=int, default=50)
a = p.parse_args()
SEEDS = a.seeds or ([42, 101, 202, 303, 404] if a.full else [42, 101, 202])
KS = a.k or ([0, 5, 10, 20, 40, 80] if a.full else [0, 20, 80])
N_EVAL = a.instances

import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def shap_class1(sv):
    """shap versions return a list per class, or an (n, d, 2) array, or an (n, d) array."""
    if isinstance(sv, list): return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def lime_attributions(model, X_train, X_eval, d, seed):
    ex = LimeTabularExplainer(X_train, mode="classification", discretize_continuous=True,
                              kernel_width=0.75 * np.sqrt(d), random_state=seed, feature_selection="none")
    out = np.zeros((len(X_eval), d))
    for i, x in enumerate(X_eval):
        e = ex.explain_instance(x, model.predict_proba, labels=(1,), num_features=d,
                                num_samples=500)
        for j, w in e.as_map()[1]:
            out[i, j] = w
    return out


rows = []
print("A. shap Exact explainer (independent masker) vs our TreeSHAP, same forest and background")
for k_irr in (0, 5):
    X, y, _ = R.generate_dataset(600, k_irr, 1.0, 0.0, 42)
    rf = R.FastRandomForestClassifier(40, 6, 8, "sqrt", 42).fit(X[:450], y[:450])
    bg, test = X[:80], X[450:455]
    ours = R.compute_treeshap_attributions(rf, test, bg)
    masker = shap.maskers.Independent(bg, max_samples=80)
    lib = shap.explainers.Exact(lambda Z: rf.predict_proba(Z)[:, 1], masker)(test).values
    diff = float(np.abs(lib - ours).max())
    print(f"   k_irr={k_irr}: max |difference| = {diff:.2e}")
    rows.append(dict(check="A_exact", k_irr=k_irr, seed=42, max_abs_diff=diff))

print("\nB/C. End-to-end comparison (mean over seeds)")
for k_irr in KS:
    for seed in SEEDS:
        t0 = time.time(); d = 5 + k_irr
        X, y, _ = R.generate_dataset(600, k_irr, 1.0, 0.0, seed)
        Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]
        Xev, Xbg = Xte[:N_EVAL], Xtr[:80]
        rf = R.FastRandomForestClassifier(40, 6, 8, "sqrt", seed).fit(Xtr, ytr)

        # ---- B: SHAP ----
        sk = RandomForestClassifier(40, max_depth=6, min_samples_leaf=8, random_state=seed).fit(Xtr, ytr)
        te = shap.TreeExplainer(sk, data=Xbg, feature_perturbation="interventional", model_output="probability")
        phi_lib = shap_class1(te.shap_values(Xev, check_additivity=False))
        err_lib = np.abs(phi_lib.sum(1) - (sk.predict_proba(Xev)[:, 1] - sk.predict_proba(Xbg)[:, 1].mean())).max()
        ev_lib = R.evaluate_explanations(phi_lib)
        phi_our = R.compute_treeshap_attributions(rf, Xev, Xbg)
        ev_our = R.evaluate_explanations(phi_our)
        # background shift for stability, as in the main experiments
        cos_lib, jac_lib = R.compute_explanation_stability(
            phi_lib, shap_class1(shap.TreeExplainer(sk, data=Xtr[40:120], feature_perturbation="interventional",
                                                    model_output="probability").shap_values(Xev, check_additivity=False)))
        cos_our, jac_our = R.compute_explanation_stability(phi_our, R.compute_treeshap_attributions(rf, Xev, Xtr[40:120]))

        # ---- C: LIME on our forest ----
        l1 = lime_attributions(rf, Xtr, Xev, d, seed + 1000)
        l2 = lime_attributions(rf, Xtr, Xev, d, seed + 2000)
        ev_ll = R.evaluate_explanations(l1)
        cos_ll, jac_ll = R.compute_explanation_stability(l1, l2)
        ours = R.TabularLIMEExplainer(Xtr, kernel_scale=0.75)
        o1 = ours.explain_batch(Xev, rf.predict_proba, num_samples=500, seed=seed + 1000)
        o2 = ours.explain_batch(Xev, rf.predict_proba, num_samples=500, seed=seed + 2000)
        ev_lo = R.evaluate_explanations(o1)
        cos_lo, jac_lo = R.compute_explanation_stability(o1, o2)

        rows.append(dict(check="BC", k_irr=k_irr, seed=seed,
                         auc_ours=R.compute_classification_metrics(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])[2],
                         auc_sklearn=roc_auc_score(yte, sk.predict_proba(Xte)[:, 1]),
                         shap_lib_local_rec=ev_lib["local_recovery"], shap_ours_local_rec=ev_our["local_recovery"],
                         shap_lib_samr=ev_lib["samr"], shap_ours_samr=ev_our["samr"],
                         shap_lib_cos=cos_lib, shap_ours_cos=cos_our, shap_lib_jac=jac_lib, shap_ours_jac=jac_our,
                         shap_lib_additivity_err=err_lib,
                         lime_lib_local_rec=ev_ll["local_recovery"], lime_ours_local_rec=ev_lo["local_recovery"],
                         lime_lib_samr=ev_ll["samr"], lime_ours_samr=ev_lo["samr"],
                         lime_lib_cos=cos_ll, lime_ours_cos=cos_lo, lime_lib_jac=jac_ll, lime_ours_jac=jac_lo))
        print(f"   k_irr={k_irr:2d} seed={seed}  done in {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/library_comparison.csv", index=False)
bc = df[df.check == "BC"].drop(columns=["check", "seed", "max_abs_diff"], errors="ignore")
pd.set_option("display.width", 220)
print("\nSHAP: library (scikit-learn forest) vs ours (our forest), percentages")
m = bc.groupby("k_irr").mean() * 100
print(m[["shap_lib_local_rec", "shap_ours_local_rec", "shap_lib_samr", "shap_ours_samr", "shap_lib_cos", "shap_ours_cos",
         "shap_lib_jac", "shap_ours_jac"]].round(1).to_string())
print("\nLIME: library vs ours, both on OUR forest, percentages")
print(m[["lime_lib_local_rec", "lime_ours_local_rec", "lime_lib_samr", "lime_ours_samr", "lime_lib_cos", "lime_ours_cos",
         "lime_lib_jac", "lime_ours_jac"]].round(1).to_string())
print("\nmax additivity error of shap library:", f"{bc.shap_lib_additivity_err.max():.1e}")
print("Saved results/library_comparison.csv")
