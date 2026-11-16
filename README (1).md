<!-- # Do Irrelevant Features Mislead SHAP and LIME?
DS5622 Explainable AI, Track 1. Thousif (142301032), Prasanna (142301027).

Random Forest, exact interventional TreeSHAP and tabular LIME are all implemented from scratch with NumPy.

```
pip install -r requirements.txt
python3 run_experiments.py                 # 4 experiments x 5 seeds, ~10 min on one core -> results/*.csv, results/fig*.png
python3 run_experiments.py --figures-only  # redraw figures from results/*.csv (seconds)
python3 experiment.py --demo               # ~25 s class demo (1 seed, 25 explained points)
python3 validate_implementation.py         # brute-force Shapley check, noise-split share, sklearn comparison
```
Seeds: 42, 101, 202, 303, 404. Runtimes in Table 3 depend on the machine; everything else is deterministic. -->

# Do Irrelevant Features Mislead SHAP and LIME?
DS5622 Explainable AI, Track 1. Thousif (142301032), Prasanna (142301027).

Random Forest, exact interventional TreeSHAP and tabular LIME are all implemented from scratch with NumPy.

```
pip install -r requirements.txt
python3 run_experiments.py                 # 4 experiments x 5 seeds, ~10 min on one core -> results/*.csv, results/fig*.png
python3 run_experiments.py --figures-only  # redraw figures from results/*.csv (seconds)
python3 experiment.py --demo               # ~25 s class demo (1 seed, 25 explained points)
python3 validate_implementation.py         # brute-force Shapley check, noise-split share, sklearn comparison
pip install shap lime scikit-learn
python3 compare_with_libraries.py          # cross-check vs the shap and lime packages (--full for all k_irr and 5 seeds)
```
Seeds: 42, 101, 202, 303, 404. Runtimes in Table 3 depend on the machine; everything else is deterministic.
