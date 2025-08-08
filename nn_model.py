from sklearn.neural_network import MLPRegressor
import numpy as np

class NNEnsemble:
    def __init__(self, n_models=3, random_state=None):
        self.models = []
        self.n_models = n_models
        self.random_state = random_state

    def fit(self, X, y):
        if hasattr(X, 'columns'):  # Convert DataFrame to NumPy array if needed
            X = X.values
        for i in range(self.n_models):
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                alpha=1e-3,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                random_state=self.random_state + i if self.random_state is not None else None,
                solver='adam',
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                tol=1e-4,
                verbose=0,
                batch_size=16
            )
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        if hasattr(X, 'columns'):  # Convert DataFrame to NumPy array if needed
            X = X.values
        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)
