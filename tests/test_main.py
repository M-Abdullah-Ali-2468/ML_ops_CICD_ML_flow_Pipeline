from sklearn.datasets import load_iris

def test_data_loading():
    X, y = load_iris(return_X_y=True)
    assert X.shape[0] > 0
    assert y.shape[0] > 0


def test_model_training():
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    assert model is not None