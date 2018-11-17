# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import spatial
from scipy.optimize import minimize

from .inference import Inference

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    # 実行環境がJupyter Notebookのとき，tqdm_notebookをimportする
    from tqdm import tqdm_notebook as tqdm
else:
    # それ以外のときは，tqdmをimportする
    from tqdm import tdqm


class Optimise(object):
    """
    GPのハイパーパラメータ最適化
    CVEに対してはグリッドサーチ，BFEに対してはグリッドサーチと準ニュートン法が実装済

    Parameters
    ----------
    train_X : array-like, shape = (n_sample, n_features)
        input data

    train_y : array-like, shape = (n_sample, 1)
        output data
    """

    def __init__(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        self.n_train = len(train_X)
        self.beta = 0.5
        self.sigma_n2 = 1.
        self.sigma_f2 = 1.
        self.hyper_params = [self.beta, self.sigma_n2, self.sigma_f2]
        self.hp_search_log = []
        self.param_log = []
        self.bfe_log = []
        self.PARAMS_BOUND = [[1e-7, 1e+2], [1e-12, 1e+4], [1e-8, 1e+8]]
        self.beta_list = None
        self.sigma_n2_list = None
        self.sigma_f2_list = None


    def set_hyper_param(self, beta, sigma_n2, sigma_f2):
        """
        ハイパーパラメータの設定

        Parameters
        ----------
        beta : float
            ガウスカーネルの幅の逆数．全ての入力次元に対して同じ値を持つ．

        sigma_n2 : float
            ノイズの分散．y = f + \\epsilon, \\epsilon ~ N(0, \\sigma_n2)．

        sigma_f2 : float
            信号分散．
        """
        self.beta = beta
        self.sigma_n2 = sigma_n2
        self.sigma_f2 = sigma_f2
        self.hyper_params = [beta, sigma_n2, sigma_f2]
    
    def optimise(self, criteria, method, signal_variance):
        """
        最適化の実行

        Parameters
        ----------
        criteria : str
            最適化基準(目的関数)
            'CVE': 交差検証誤差(Cross Validation Error) (RMSE)
            'BFE': ベイズ自由エネルギー(Bayesian Free Energy) (Negative Log Marginal Likelihood)

        method : str
            最適化基準(目的関数)の最小化方法
            'gridsearch': グリッドサーチ
            'L-BFGS-B': 準ニュートン法の1種

        signal_variance : bool
            False: k(x) = exp(- \\beta |x|^2)
            True: k(x) = \\sigma_f2 * exp(- \\beta |x|^2)
        """
        if method == 'gridsearch':
            self.gridsearch(criteria, signal_variance)

        elif method == 'L-BFGS-B':
            self.lbfgsb()

        return self.hyper_params

    def lbfgsb(self):
        """
        L-BFGS-Bによる非線形最適化
        scipy.optimize.minimize
        """
        res = minimize(fun=self._bfe, x0=np.array(self.hyper_params), method='L-BFGS-B', jac=self._d_bfe,
                       bounds=self.PARAMS_BOUND)
        # self.hp_search_log.append([res.x, res.fun])
        self.bfe = res.fun
        self.set_hyper_param(res.x[0], res.x[1], res.x[2])

    def gridsearch(self, criteria, signal_variance):
        """
        グリッドサーチ
        2段階のグリッドサーチを行う．
        step 1では広い範囲を荒く探索し最小値をとるパラメータを見つける．
        step 2ではstep 1で見つけたパラメータ周辺の狭い範囲で細かく探索する．

        Parameters
        ----------
        criteria : str

        signal_variance : str
        """
        if criteria == 'CVE':
            evaluation = self._evaluation_cve
        elif criteria == 'BFE':
            evaluation = self._evaluation_bfe

        if signal_variance:
            self.beta_list = np.logspace(np.log10(self.PARAMS_BOUND[0][0]), np.log10(self.PARAMS_BOUND[0][1]),
                                         np.log10(self.PARAMS_BOUND[0][1]) - np.log10(self.PARAMS_BOUND[0][0]) + 1)
            self.sigma_n2_list = np.logspace(np.log10(self.PARAMS_BOUND[1][0]), np.log10(self.PARAMS_BOUND[1][1]),
                                         np.log10(self.PARAMS_BOUND[1][1]) - np.log10(self.PARAMS_BOUND[1][0]) + 1)
            self.sigma_f2_list = np.logspace(np.log10(self.PARAMS_BOUND[2][0]), np.log10(self.PARAMS_BOUND[2][1]),
                                         np.log10(self.PARAMS_BOUND[2][1]) - np.log10(self.PARAMS_BOUND[2][0]) + 1)

            # step 1
            grid = np.array([[[evaluation(b, s, sf) for s in self.sigma_n2_list]
                                for b in self.beta_list]
                                for sf in tqdm(self.sigma_f2_list, desc='step 1')])
            min_idx = np.unravel_index(np.argmin(grid), grid.shape)
            sf1 = self.sigma_f2_list[min_idx[0]]
            b1 = self.beta_list[min_idx[1]]
            s1 = self.sigma_n2_list[min_idx[2]]

            # step 2
            self.beta_list = np.logspace(max(np.log10(self.PARAMS_BOUND[0][0]), np.log10(b1) - 1),
                                         min(np.log10(self.PARAMS_BOUND[0][1]), np.log10(b1) + 1),
                                         51)
            self.sigma_n2_list = np.logspace(max(np.log10(self.PARAMS_BOUND[1][0]), np.log10(s1) - 1),
                                             min(np.log10(self.PARAMS_BOUND[1][1]), np.log10(s1) + 1),
                                             51)
            self.sigma_f2_list = np.logspace(max(np.log10(self.PARAMS_BOUND[2][0]), np.log10(sf1) - 1),
                                             min(np.log10(self.PARAMS_BOUND[2][1]), np.log10(sf1) + 1),
                                             51)
            self.grid = np.array([[[evaluation(b, s, sf) for s in self.sigma_n2_list]
                                for b in self.beta_list]
                                for sf in tqdm(self.sigma_f2_list, desc='step 1')])
            min_idx = np.unravel_index(np.argmin(self.grid), self.grid.shape)
            sigma_f2_opt = self.sigma_f2_list[min_idx[0]]
            beta_opt = self.beta_list[min_idx[1]]
            sigma_n2_opt = self.sigma_n2_list[min_idx[2]]

            self.set_hyper_param(beta_opt, sigma_n2_opt, sigma_f2_opt)

        else:
            self.beta_list = np.logspace(np.log10(self.PARAMS_BOUND[0][0]), np.log10(self.PARAMS_BOUND[0][1]),
                                         np.log10(self.PARAMS_BOUND[0][1]) - np.log10(self.PARAMS_BOUND[0][0]) + 1)
            self.sigma_n2_list = np.logspace(np.log10(self.PARAMS_BOUND[1][0]), np.log10(self.PARAMS_BOUND[1][1]),
                                         np.log10(self.PARAMS_BOUND[1][1]) - np.log10(self.PARAMS_BOUND[1][0]) + 1)

            # step 1
            grid = np.array([[evaluation(b, s, 1.) for s in self.sigma_n2_list]
                                for b in tqdm(self.beta_list, desc='step 1')])
            min_idx = np.unravel_index(np.argmin(grid), grid.shape)
            b1 = self.beta_list[min_idx[0]]
            s1 = self.sigma_n2_list[min_idx[1]]

            # step 2
            self.beta_list = np.logspace(max(np.log10(self.PARAMS_BOUND[0][0]), np.log10(b1) - 1),
                                         min(np.log10(self.PARAMS_BOUND[0][1]), np.log10(b1) + 1),
                                         51)
            self.sigma_n2_list = np.logspace(max(np.log10(self.PARAMS_BOUND[1][0]), np.log10(s1) - 1),
                                         min(np.log10(self.PARAMS_BOUND[1][1]), np.log10(s1) + 1),
                                         51)
            self.grid = np.array([[evaluation(b, s, 1.) for s in self.sigma_n2_list]
                                     for b in tqdm(self.beta_list, desc='step 2')])
            min_idx = np.unravel_index(np.argmin(self.grid), self.grid.shape)
            beta_opt = self.beta_list[min_idx[0]]
            sigma_n2_opt = self.sigma_n2_list[min_idx[1]]

            self.set_hyper_param(beta_opt, sigma_n2_opt, 1.)

    def _evaluation_cve(self, b, s, sf):
        return self._cve(Inference(b, s, sf))

    def _evaluation_bfe(self, b, s, sf):
        return self._bfe([b, s, sf])

    def _cve(self, infr, fold=10):
        """
        k-fold交差検証

        Parameters
        ----------
        infr : Inference object

        fold : int, optional (default : 10)
            fold数

        Returns
        ----------
        cve_ : float
            CVE
        """
        n_valid = self.n_train // fold
        np.random.seed(0)
        idx = list(np.random.permutation(np.arange(self.n_train)))
        idx_set = set(idx)
        rmses = []

        for i in range(fold):
            idx_valid = idx[n_valid * i: n_valid * (i + 1)]
            idx_train = list(idx_set - set(idx_valid))
            train_X, train_y = self.train_X[idx_train], self.train_y[idx_train]
            valid_X, valid_y = self.train_X[idx_valid], self.train_y[idx_valid]
            pred_y = infr.predict_given_hp(train_X, train_y, valid_X)
            rmses.append(self._rmse(valid_y, pred_y))

        cve_ = np.array(rmses).mean()
        return cve_

    def _rmse(self, test_y, pred_y):
        """
        平方根平均二乗誤差(Root Mean Square Error, RMSE)

        Parameters
        ----------
        test_y : array-like, shape = (n_sample, 1)
            予測値

        pred_y : array-like, shape = (n_sample, 1)
            真値

        Returns
        ----------
        rmse : float
            RMSE
        """
        rmse_ = np.sqrt(np.mean((test_y - pred_y) ** 2))
        return rmse_

    def _bfe(self, params):
        """
        Bayesian Free Energy(FBE, negative log marginal likelihood)

        Parameters
        ----------
        params : list, [beta, sigma_n2, sigma_f2]
            hyper parameters
        """
        K = params[2] * np.exp(-params[0] * spatial.distance.cdist(self.train_X, self.train_X) ** 2)
        Ky = K + params[1] * np.eye(self.n_train)
        L = linalg.cho_factor(Ky)
        # print(np.sum(Ky))
        alpha = linalg.cho_solve(L, self.train_y)
        # print(np.matmul(self.train_y.T, alpha)[0][0])
        return 0.5 * (np.matmul(self.train_y.T, alpha)[0][0] + np.sum(np.log(np.diag(L[0]))) * 2 
                      + self.n_train * np.log(2*np.pi))

    def _d_bfe(self, params):
        """
        Partial derivatives of BFE

        Parameters
        ----------
        params : list, [beta, sigma_n2, sigma_f2]
            hyper parameters
        """
        D = spatial.distance.cdist(self.train_X, self.train_X) ** 2
        K_ = np.exp(-params[0] * D)
        K = params[2] * K_
        Ky = K + params[1] * np.eye(self.n_train)
        L = linalg.cho_factor(Ky)
        alpha = linalg.cho_solve(L, self.train_y)
        alphaalpha = np.matmul(alpha, alpha.T)
        A = alphaalpha - linalg.inv(Ky)
        return -0.5 * np.array([np.trace(np.matmul(A, -D * K)), np.trace(A), np.trace(np.matmul(A, K_))])
