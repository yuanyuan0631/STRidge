# 利用 STRidge 实现偏微分方程未知项的估计
import numpy as np
import sys


class STRidge():
    def __init__(self, R0, Ut, lambda1, train_it, lam=1e-5, d_tol=1, maxit=100,
                 STR_iters=10, l0_penalty=None, normalize=2, split=0.8,
                 print_best_tol=False, tol=None, l0_penalty_0=None):

        try:
            self.R0 = R0.detach().cpu().numpy()
        except AttributeError:
            self.R0 = np.array(R0)

        try:
            self.Ut = Ut.detach().cpu().numpy()
        except AttributeError:
            self.Ut = np.array(Ut)

        try:
            self.lambda1 = lambda1.detach().cpu().numpy()
        except AttributeError:
            self.lambda1 = np.array(lambda1)

        self.lam = lam
        self.d_tol = d_tol
        self.maxit = maxit
        self.STR_iters = STR_iters
        self.l0_penalty = l0_penalty
        self.normalize = normalize
        self.split = split
        self.print_best_tol = print_best_tol

        self.l0_penalty_0 = l0_penalty_0

        self.it = train_it

        if self.it != 0:
            print("Warning! The parameter `tol` is needed!")
            self.tol = tol

    def STRidge(self, X0, y, lam, maxit, tol, Mreg, normalize=2):

        n, d = X0.shape
        X = np.zeros((n, d), dtype=np.complex64)

        if normalize != 0:
            Mreg = np.zeros((d, 1))
            for i in range(d):
                Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
                X[:, i] = Mreg[i] * X0[:, i]

        else:
            X = X0

        w = self.lambda1 / Mreg

        biginds = np.where(abs(w) > tol)[0]
        num_relevant = d

        for j in range(maxit):

            smallinds = np.where(abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]

            if num_relevant == len(new_biginds):
                break
            else:
                num_relevant = len(new_biginds)

            if len(new_biginds) == 0:
                if j == 0:
                    if normalize != 0:
                        return np.multiply(Mreg, w)

                    else:
                        return w

                else:
                    break

            biginds = new_biginds
            w[smallinds] = 0

            if lam != 0:
                w[biginds] = np.linalg.lstsq(
                    X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
            else:
                w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]

        if biginds != []:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]

        if normalize != 0:
            return np.multiply(Mreg, w)

        else:
            return w

    def fit(self):

        n, d = self.R0.shape
        R = np.zeros_like(self.R0)

        if self.normalize != 0:
            Mreg = np.zeros((d, 1))
            for i in range(0, d):
                Mreg[i] = 1.0 / (np.linalg.norm(self.R0[:, i], self.normalize))
                R[:, i] = Mreg[i] * self.R0[:, i]
            self.normalize = 0
        else:
            R = self.R0
            Mreg = np.ones((d, 1)) * d
            self.normalize = 2

        train = np.random.choice(n, int(n * self.split), replace=False)
        val = [i for i in np.arange(n) if i not in train]

        Train_R = R[train, :]
        Val_R = R[val, :]
        Train_y = self.Ut[train, :]
        Val_y = self.Ut[val, :]

        d_tol = float(self.d_tol)
        if self.it == 0:
            self.tol = d_tol

        w_best = self.lambda1 / Mreg

        err_f = np.mean((Val_y - np.dot(Val_R, w_best))**2)

        if self.l0_penalty is None and self.it == 0:
            self.l0_penalty_0 = err_f
            l0_penalty = self.l0_penalty_0
        elif self.l0_penalty is None:
            print("Warning! l0_penalty is None! Please input the parameter `l0_penalty_0` from the previous training")
            l0_penalty = self.l0_penalty_0

        err_lambda = l0_penalty * np.count_nonzero(w_best)
        err_best = err_f + err_lambda
        tol_best = 0

        for iter in range(self.maxit):
            w = self.STRidge(Train_R, Train_y, self.lam, self.STR_iters, self.tol,
                             Mreg, normalize=self.normalize)

            err_f = np.mean((Val_y - np.dot(Val_R, w))**2)

            err_lambda = l0_penalty * np.count_nonzero(w)

            err = err_f + err_lambda

            if err <= err_best:
                err_best = err
                w_best = w
                tol_best = self.tol
                self.tol = self.tol + d_tol

            else:
                self.tol = max([0, self.tol - 2 * d_tol])
                d_tol = d_tol / 1.618
                self.tol = self.tol + d_tol

        if self.print_best_tol:
            print("Err best: %e, err_f: %e, err_lambda: %e," % (err_best, err_f, err_lambda))
            print("Optimal tolerance:", tol_best)
            print("\n")

        return np.real(np.multiply(Mreg, w_best))


sys.modules[__name__] = STRidge
