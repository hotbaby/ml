# encoding: utf8

import copy
import time
import numpy as np
from sklearn.metrics import roc_auc_score


class MyBaggingClassifier(object):
    """Bagging classifier."""

    def __init__(self, base_estimator, n_estimators=1000, random_state=42, max_samples=1.0, n_jobs=12):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = 1.0
        self.estimators = []

    def fit(self, X, y, val_set=None):
        self.estimators = []
        n_samples = int(len(X) * self.max_samples)

        for _ in range(self.n_estimators):
            estimator = copy.deepcopy(self.base_estimator)
            rng = np.random.RandomState(self.random_state)
            samples = []

            tm = time.time()
            for i in range(n_samples):
                samples.append(rng.choice(n_samples))

            X_samples = X[samples]
            y_samples = y[samples]
            estimator.fit(X_samples, y_samples)

            self.estimators.append(estimator)

            if val_set:
                X_val, y_val = val_set
                auc_score = roc_auc_score(y_val, self.predict_proba(X_val)[:, 1])
                msg = 'fit %s estimators, auc: %.6f, elapse %.2f seconds, ' % (len(self.estimators),
                                                                               auc_score,
                                                                               time.time() - tm)
            else:
                msg = 'fit %s estimators, elapse %.2f seconds ' % (len(self.estimators), time.time() - tm)
            print(msg)

    def predict_proba(self, X):
        assert len(self.estimators) > 0, 'Please train the model first!'

        y_pred = np.zeros((len(X), 2))

        for estimator in self.estimators:
            y_pred += estimator.predict_proba(X)

        return y_pred / len(self.estimators)
