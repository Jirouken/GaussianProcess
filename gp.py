# -*- coding: utf-8 -*-
from .inference import Inference
from .optimise import Optimise


class Regression(object):
    """
    ガウス過程回帰(Gaussian Process Regression, GPR)
    参考：Gaussian Process for Machine learning chap.2
        (http://www.gaussianprocess.org/gpml/chapters/RW2.pdf)
    Multi‐input Single‐output,
    Gaussian (RBF) kernel k(x) = \\sigma_f2 * exp(- \\beta |x|^2).

    Attributes
    ----------
    beta : float
        ガウスカーネルの幅の逆数．全ての入力次元に対して同じ値を持つ．

    sigma_n2 : float
        ノイズの分散．y = f + \\epsilon, \\epsilon ~ N(0, \\sigma_n2)．

    sigma_f2 : float
        信号分散．

    hyper_params : list
        [beta, sigma_n2, sigma_f2]
    """

    def __init__(self):
        self.beta = 0.5
        self.sigma_n2 = 1.
        self.sigma_f2 = 1.
        self.hyper_params = [self.beta, self.sigma_n2, self.sigma_f2]
        self.PARAMS_BOUND = [[1e-7, 1e+2], [1e-12, 1e+4], [1e-8, 1e+8]]
        self.train_X = None
        self.train_y = None
        self.n_train = None

    def fit(self, train_X, train_y, optimization=True, criteria='CVE', method='gridsearch', signal_variance=False):
        """
        訓練データとハイパーパラメータ最適化についての設定

        Parameters
        ----------
        train_X : array-like, shape = (n_sample, n_features)
            input data

        train_y : array-like, shape = (n_sample, 1)
            output data

        optimization : bool, optional (default: True)
            True: ハイパーパラメータの最適化を行う
            False: 与えられたハイパーパラメータで学習を行う

        criteria : str, optional (default: 'CVE')
            最適化基準(目的関数)
            'CVE': 交差検証誤差(Cross Validation Error) (RMSE)
            'BFE': ベイズ自由エネルギー(Bayesian Free Energy) (Negative Log Marginal Likelihood)

        method : str, optional (default: 'gridsearch')
            最適化基準(目的関数)の最小化方法
            'gridsearch': グリッドサーチ
            'L-BFGS-B': 準ニュートン法の1種

        signal_variance : bool, optional (default: False)
            False: k(x) = exp(- \\beta |x|^2)
            True: k(x) = \\sigma_f2 * exp(- \\beta |x|^2)
        """
        assert criteria == 'CVE' or criteria == 'BFE', '最適化基準criteriaは"CVE"か"BFE"しか選択できません．'

        if criteria == 'CVE':
            assert method == 'gridsearch', \
                '最適化基準criteriaが"CVE"の時は最適化方法methodに"gridsearch"しか選択できません．'

        if optimization == True:
            # if the optimization method is designated
            self.opt = Optimise(train_X, train_y)
            self.beta, self.sigma_n2, self.sigma_f2 = self.opt.optimise(criteria, method, signal_variance)
            self.set_hyper_param(self.beta, self.sigma_n2, self.sigma_f2)

        self.infr = Inference(self.beta, self.sigma_n2, self.sigma_f2)
        self.infr._train(train_X, train_y)

    def predict(self, test_X, cov=False):
        """
        テストデータtest_Xに対して予測を行う．

        Parameters
        ----------
        test_X : array-like, shape = (n_sample, n_features)
            input data

        cov : bool, optional (default: False)
            True: 予測分布の分散共分散行列を返す

        Returns
        ----------
        f_mean : array, shape = (n_sample, 1)
            テストデータtest_Xに対する予測平均

        f_cov : array, shape = (n_sample, n_sample)
            予測分布の分散共分散行列
        """
        f_mean, f_cov = self.infr.inference(test_X)
        if cov:
            return f_mean, f_cov
        else:
            return f_mean

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

    def sample_from_posterior(self, test_X, size):
        """
        GP事後分布からのサンプリング
        """
        f_mean, f_cov = self.infr.inference(test_X)
        samples = np.random.multivariate_normal(f_mean[:, 0], f_cov, size)
	return samples

    def sample_from_prior(self, test_X, size):
        """
        GP事前分布からのサンプリング
        """
        infr = Inference(self.beta, self.sigma_n2, self.sigma_f2)
        K_tt = infr._cov(test_X, test_X)
        f_mean = np.zeros(len(test_X))
        samples = np.random.multivariate_normal(f_mean, K_tt, size)
        return samples
