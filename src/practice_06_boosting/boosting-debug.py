from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def get_predictions_tensor(self, probas):
        if len(probas.shape) == 1:
            return np.hstack([
                (1. - probas).reshape(-1, 1),
                probas.reshape(-1, 1),
            ])
        elif len(probas.shape) == 2:
            return np.hstack([
                (1. - probas), probas,
            ])
        else:
            raise AssertionError

    def fit_new_base_model(self, x, y, predictions):
        boostraped_samples = np.random.choice(
            np.arange(x.shape[0]),
            int(self.subsample * x.shape[0]),
            replace=True
        )
        boostraped_x = x[boostraped_samples]
        boostraped_y = y[boostraped_samples]
        boostraped_pred = predictions[boostraped_samples]

        shifts = -self.loss_derivative(boostraped_y, boostraped_pred)

        new_model = self.base_model_class(**self.base_model_params)
        new_model.fit(boostraped_x, shifts)
        new_model_pred = new_model.predict(boostraped_x)

        best_gamma = self.find_optimal_gamma(boostraped_y, boostraped_pred, new_model_pred)

        self.gammas.append(best_gamma)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        last_valid_loss = 1e10
        iterations_on_plateu = 0
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.models[-1].predict(x_train)
            if self.early_stopping_rounds is not None:
                valid_predictions += self.models[-1].predict(x_valid)
                valid_loss = self.loss_fn(y_valid, valid_predictions)
                if valid_loss >= last_valid_loss:
                    iterations_on_plateu += 1
                else:
                    iterations_on_plateu = 0
                if iterations_on_plateu >= self.early_stopping_rounds:
                    break
                last_valid_loss = valid_loss

        if self.plot:
            pass

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += model.predict(x) * gamma * self.learning_rate
        predictions = self.sigmoid(predictions)
        return self.get_predictions_tensor(predictions)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass


### DEBUG

x = load_npz('x.npz')
y = np.load('y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1337)
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=1337)

boosting = Boosting()

boosting.fit(x_train, y_train, x_valid, y_valid)

assert len(boosting.models) == boosting.n_estimators
assert len(boosting.gammas) == boosting.n_estimators

assert boosting.predict_proba(x_test).shape == (x_test.shape[0], 2)

print(f'Train ROC-AUC {boosting.score(x_train, y_train):.4f}')
print(f'Valid ROC-AUC {boosting.score(x_valid, y_valid):.4f}')
print(f'Test ROC-AUC {boosting.score(x_test, y_test):.4f}')