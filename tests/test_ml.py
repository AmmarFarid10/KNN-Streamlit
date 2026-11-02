import numpy as np
import pytest
from ml import classifier


def test_scaler_uses_training_data_only():
    model = classifier("KNN", n_neighbors=1)
    x = np.array([[0., 100.], [2., 200.], [4., 300.]])
    model.fit(x, [0, 0, 1])
    mean = model[0].mean_.copy()
    model.predict([[10000., 10000.]])
    np.testing.assert_allclose(mean, [2., 200.])
    np.testing.assert_array_equal(mean, model[0].mean_)


@pytest.mark.parametrize("name", ["SVM", "KNN", "Random Forest"])
def test_classifier_predictions(name):
    model = classifier(name, **({"n_neighbors": 1} if name == "KNN" else {}))
    model.fit([[0], [1], [10], [11]], [0, 0, 1, 1])
    assert list(model.predict([[0], [11]])) == [0, 1]


def test_unknown_classifier_rejected():
    with pytest.raises(ValueError):
        classifier("unsupported")
