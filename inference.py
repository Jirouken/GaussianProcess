# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import spatial


class Inference(object):

    def __init__(self, beta, sigma_n2, sigma_f2):
        self.beta = beta
        self.sigma_n2 = sigma_n2
        self.sigma_f2 = sigma_f2
        self.train_X = None
        self.train_y = None
        self.n_train = None

    def predict_given_hp(self, train_X, train_y, test_X):
        self._train(train_X, train_y)
        pred_y, _ = self.inference(test_X)
        return pred_y

    def _train(self, train_X, train_y):
        """
        Calculate alpha(GPML, p.19)

        Parameters
        ----------
        train_X : array-like, shape = (n_sample, n_features)
            input data

        train_y : array-like, shape = (n_sample, 1)
            output data
        """
        self.train_X = train_X
        self.train_y = train_y
        self.n_train = len(train_y)
        # covariance matrix
        K = self._cov(self.train_X, self.train_X)
        if self.sigma_n2 > 1e-16:
            # cholesky decomposition
            # LL^T = K + sigma_n^2 * I
            # alpha_ = (K + sigma_n^2 * I)^(-1) y
            self.L = linalg.cho_factor(K + self.sigma_n2 * np.eye(self.n_train))
            self.alpha = linalg.cho_solve(self.L, self.train_y)
        else:
            # cannot perform cholesky decomposition
            self.C_inv = np.linalg.inv(K + self.sigma_n2 * np.eye(self.n_train))
            self.alpha = np.matmul(self.C_inv, self.train_y)

    def inference(self, test_X):
        """
       Predict y for test_X

       Parameters
       ----------
       test_X : array-like, shape = (n_sample, n_features)
           input data

       cov : bool, optional (default: False)
           If true, return covariance matrix of predictive values.

       Returns
       ----------
       f_mean : array, shape = (n_sample, 1)
           predictive mean

       f_cov : array, shape = (n_sample, n_sample)
           covariance matrix of predictive values
       """
        K_t = self._cov(test_X, self.train_X)
        K_tt = self._cov(test_X, test_X)
        f_mean = np.matmul(K_t, self.alpha)

        # predictive variance
        if self.sigma_n2 > 1e-16:
            # cholesky decomposition
            # V = (K + sigma_n^2 * I)^(-1) K_t^T
            V = linalg.cho_solve(self.L, K_t.T)
            f_cov = K_tt - np.matmul(K_t, V)
            return f_mean, f_cov
        else:
            # cannot perform cholesky decomposition
            V = np.matmul(self.C_inv, K_t.T)
            f_cov = K_tt - np.matmul(K_t, V)
            return f_mean, f_cov

    def _cov(self, X, X_):
        """
        Covariance Matrix

        Parameters
        ----------
        X, X_ : array-like, shape = (n_sample, n_features)
            design matrix

        Returns
        ----------
        K : array, shape=(n_sample, n_sample)
        """
        K = self.sigma_f2 * np.exp(-self.beta * spatial.distance.cdist(X, X_) ** 2)
        return K
