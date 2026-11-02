"""Pure training utilities kept separate from Streamlit rendering."""
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def classifier(name: str, **parameters: Any) -> Pipeline:
    if name == "SVM":
        estimator = SVC(**parameters)
    elif name == "KNN":
        estimator = KNeighborsClassifier(**parameters)
    elif name == "Random Forest":
        estimator = RandomForestClassifier(random_state=1, **parameters)
    else:
        raise ValueError("Unknown classifier")
    # Scaling is fitted ONLY when fitting the training set; avoids test leakage.
    return make_pipeline(StandardScaler(), estimator)
