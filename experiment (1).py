#!/usr/bin/env python3
"""
DS5622 Explainable AI Course Project (Track 1)
Do Irrelevant Features Mislead SHAP and LIME? An Experimental Investigation
Authors: Thousif (142301032) & Prasanna (142301027)

Usage:
  python3 experiment.py --demo   # quick walkthrough: 1 seed, 25 explained points, ~25 s
  python3 experiment.py --full   # same as run_experiments.py (5 seeds, ~10 min)

All the real code (data generator, Random Forest, TreeSHAP, LIME, metrics) lives in
run_experiments.py; this file only calls it.
"""
import argparse
import pandas as pd
from run_experiments import (K_INF, generate_dataset, FastRandomForestClassifier,
                             compute_classification_metrics, compute_treeshap_attributions,
                             TabularLIMEExplainer, evaluate_explanations,
                             compute_explanation_stability, main as run_full_suite)


def run_demo():
    print("=" * 78)
    print(" DS5622 EXPLAINABLE AI: LIVE DEMO (seed 42, 25 explained test points)")
    print(" Do Irrelevant Features Mislead SHAP and LIME?")
    print("=" * 78)
    print("True informative features: x1 (w=2.20, linear) > x2 (1.65, linear) > x3 (1.25, quadratic)")
    print("                           > x4 (0.90, sinusoidal) > x5 (0.55, linear)")
    rows = []
    for k_irr in [0, 10, 20, 40, 80]:
        X, y, _ = generate_dataset(n_samples=600, k_irr=k_irr, noise_std=1.0, seed=42)
        X_train, y_train, X_test, y_test = X[:450], y[:450], X[450:], y[450:]
        X_eval, X_bg = X_test[:25], X_train[:60]

        rf = FastRandomForestClassifier(40, 6, 8, seed=42).fit(X_train, y_train)
        _, f1, auc = compute_classification_metrics(y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

        shap1 = compute_treeshap_attributions(rf, X_eval, X_bg)
        shap2 = compute_treeshap_attributions(rf, X_eval, X_train[30:90])   # shifted background
        s_ev = evaluate_explanations(shap1)
        s_cos, _ = compute_explanation_stability(shap1, shap2)
        fx = rf.predict_proba(X_eval)[:, 1]
        efx = rf.predict_proba(X_bg)[:, 1].mean()
        err = abs(shap1.sum(axis=1) - (fx - efx)).max()

        lime = TabularLIMEExplainer(X_train, kernel_scale=0.75)
        lime1 = lime.explain_batch(X_eval, rf.predict_proba, num_samples=500, seed=1042)
        lime2 = lime.explain_batch(X_eval, rf.predict_proba, num_samples=500, seed=5042)
        l_ev = evaluate_explanations(lime1)
        l_cos, _ = compute_explanation_stability(lime1, lime2)

        rows.append({"k_irr": k_irr, "AUC %": round(auc * 100, 1), "F1 %": round(f1 * 100, 1),
                     "SHAP top5 %": round(s_ev["local_recovery"] * 100, 1),
                     "LIME top5 %": round(l_ev["local_recovery"] * 100, 1),
                     "SHAP noise %": round(s_ev["samr"] * 100, 1),
                     "LIME noise %": round(l_ev["samr"] * 100, 1),
                     "SHAP stab %": round(s_cos * 100, 1), "LIME stab %": round(l_cos * 100, 1),
                     "SHAP local-acc err": f"{err:.1e}"})
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    last = rows[-1]
    print("\nWhat this run shows (numbers read from the table above):")
    print(f" - TreeSHAP local accuracy holds to machine precision (largest error {max(float(r['SHAP local-acc err']) for r in rows):.1e}).")
    print(f" - At k_irr=80 the model still has ROC-AUC {last['AUC %']}%, but LIME puts {last['LIME noise %']}% of its")
    print(f"   attribution on noise (TreeSHAP: {last['SHAP noise %']}%) and its stability is {last['LIME stab %']}% (TreeSHAP: {last['SHAP stab %']}%).")
    print(" - One seed and 25 points are noisy; use --full for the 5-seed numbers in the report.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.full:
        import sys
        sys.argv = [sys.argv[0]]
        run_full_suite()
    else:
        run_demo()
