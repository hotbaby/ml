# encoding: utf8

import copy
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


class MyAdaBoostClassifier(object):
    """Boosting ensemble classififer."""

    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y, val_set=None):
        self.estimators = []
        self.estimator_weights = []

        n_samples = len(X)
        sample_weights = np.zeros(n_samples) + 1 / n_samples

        for i in range(self.n_estimators):
            estimator = copy.deepcopy(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weights)

            y_pred = estimator.predict(X)

            # calculate error rate
            err_rate = np.sum(sample_weights[y != y_pred])
            if err_rate > 0.5:
                print('error rate %.4f, break' % err_rate)
                break

            # update sample weights
            beta = err_rate / (1 - err_rate)
            sample_weights[y_pred == y] *= beta   # correct classification

            # normalize sample weights
            sample_weights /= np.sum(sample_weights)

            self.estimator_weights.append(np.log(1/beta))
            self.estimators.append(estimator)

            train_auc = roc_auc_score(y, self.predict_proba(X)[:, 1])
            if val_set:
                X_val, y_val = val_set
                auc = roc_auc_score(y_val, self.predict_proba(X_val)[:, 1])
                print('fit %s estimators, error rate: %.6f, log(1/beta): %.6f, train auc: %.6f val auc: %.6f' %
                      (i, err_rate, np.log(1 / beta), train_auc, auc))
            else:
                print('fit %s estimators, error rate: %.6f, log(1/beta): %.6f, train auc: %.6f' %
                      (i, err_rate, np.log(1/beta), train_auc))

    def predict_proba(self, X):
        assert len(self.estimators) > 0, 'Please train the model first!'

        y_pred = np.zeros((len(X), 2))
        for i, estimator in enumerate(self.estimators):
            y_pred += (estimator.predict_proba(X) * self.estimator_weights[i])

        return y_pred
