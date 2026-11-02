# Classifiers in Action
An interactive learning comparison of KNN, SVM and Random Forest on scikit-learn's Iris, Wine and Breast Cancer datasets. This is an educational benchmark, not a clinical decision tool.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
streamlit run app.py
```
Choose dataset/model in the sidebar. Train/test splits are stratified with a fixed seed. Scaling is part of a pipeline fitted only on training data. PCA plots visualize the full dataset and are exploratory; no PCA features feed the classifier. Scores reflect one split and must not be represented as robust comparative results. Cross-validation and a separate holdout should precede any model-selection claim.

```bash
pip install pytest
python -m pytest -q
```
The repair also replaces an obsolete Matplotlib 3D axes API. UI rendering still needs an interactive smoke test.
