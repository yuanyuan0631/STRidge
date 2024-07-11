# 利用 STRidge 实现偏微分方程未知项的估计
import numpy as np
import sys


class STRidge():
    def __init__(self, R0, Ut, lambda1, train_it, lam=1e-5, d_tol=1, maxit=100,
                 STR_iters=10, l0_penalty=None, normalize=2, split=0.8,
                 print_best_tol=False, tol=None, l0_penalty_0=None):

        # X0: 候选偏微分方程项的矩阵
        # y: 目标值(即为 u_t 项)
        # lambda1：神经网络参数传入项，判断有哪些项的索引向量
        # train_it：当前调用 STRidge 方法时 PINN 主训练函数优化轮数
        # lam：正则化参数，用于控制模型的复杂度，防止过拟合。
        # d_tol：初始化的容差
        # maxit：最大迭代次数，控制优化过程的迭代次数。
        # STR_iters：调用 STRidge 的最大迭代次数
        # l0_penalty：初始化 l0 正则化惩罚项
        # print_best_tol：是否打印最优容差和 STRidge 训练误差
        # normalize：归一化系数

        # 输入数据
        # self.R0 = R0.detach().cpu().numpy()
        # self.Ut = Ut.detach.cpu().numpy()
        try:
            self.R0 = R0.detach().cpu().numpy()
        except AttributeError:
            # 如果失败，尝试将其转换为 numpy 类型
            self.R0 = np.array(R0)

        try:
            self.Ut = Ut.detach().cpu().numpy()
        except AttributeError:
            # 如果失败，尝试将其转换为 numpy 类型
            self.Ut = np.array(Ut)

        try:
            self.lambda1 = lambda1.detach().cpu().numpy()
        except AttributeError:
            # 如果失败，尝试将其转换为 numpy 类型
            self.lambda1 = np.array(lambda1)

        # 模型参数指定
        self.lam = lam  # 正则化参数
        self.d_tol = d_tol  # 初始化的容差
        self.maxit = maxit  # STRidge 训练最大迭代次数
        self.STR_iters = STR_iters  # 每调用一次 STRidge 最大迭代次数
        self.l0_penalty = l0_penalty  # 初始化的惩罚参数
        self.normalize = normalize  # 归一化索引
        self.split = split  # 划分训练集验证集的比例
        self.print_best_tol = print_best_tol

        # 如果 l0_penalty 为 None 并且 Adam 优化器迭代轮数不为 0，需要用到该参数。该参数表示前一步的最优容差
        self.l0_penalty_0 = l0_penalty_0

        # PINN 迭代轮数
        self.it = train_it

        # 如果 Adam 优化器迭代数不为零，需要手动输入 tol 的值
        if self.it != 0:
            print("Warning! The parameter `tol` is needed!")
            self.tol = tol

    # 定义一次 STRidge 训练过程
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

        # 初始化权重矩阵 w
        w = self.lambda1 / Mreg

        # 初始化大参数索引
        biginds = np.where(abs(w) > tol)[0]  # 找到大系数对应的索引，并对这些系数予以保留
        num_relevant = d  # 初始的特征数量(初始化为 d，也就是所有特征都考虑了进去)

        # 经过漫长迭代找到所有的大系数
        for j in range(maxit):

            # 找到小于阈值 tol 的系数索引(小系数)
            smallinds = np.where(abs(w) < tol)[0]
            # 计算新的大系数索引列表
            new_biginds = [i for i in range(d) if i not in smallinds]

            # 如果大系数数量没有改变，那么就终止循环(已经找到了所有应该含有的导函数项)
            if num_relevant == len(new_biginds):
                break
            else:
                num_relevant = len(new_biginds)  # 否则更新大系数数量

            # 处理大系数为空的情况
            if len(new_biginds) == 0:
                # 首次迭代情况就发现没有列表中的偏导数项时，直接返回解归一化后的权重参数 w
                if j == 0:
                    # 如果归一化了，就返回解归一化后的系数 w
                    if normalize != 0:
                        return np.multiply(Mreg, w)

                    # 否则返回当前系数
                    else:
                        return w

                # 如果不是第一次迭代，那么就终止循环(选择完毕了)
                else:
                    break

            # 更新大系数，并将小系数(小于容差的系数)置零
            biginds = new_biginds
            w[smallinds] = 0

            # 求解最小二乘问题
            if lam != 0:
                w[biginds] = np.linalg.lstsq(
                    X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
            else:
                w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]

        if biginds != []:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]

        # 所有步骤求解完了，最终返回最小二乘法的参数 w，即 lambda。能反应出方程究竟含有哪些项
        if normalize != 0:
            return np.multiply(Mreg, w)

        else:
            return w

    # 定义 STRidge 训练过程
    def fit(self):

        # d_tol:初始化容差值
        n, d = self.R0.shape
        R = np.zeros_like(self.R0)

        # 计算归一化矩阵 Mreg，后面计算误差用得上，并进行归一化
        if self.normalize != 0:
            Mreg = np.zeros((d, 1))  # 计算第 i 个特征的归一化系数
            for i in range(0, d):
                Mreg[i] = 1.0 / (np.linalg.norm(self.R0[:, i], self.normalize))
                R[:, i] = Mreg[i] * self.R0[:, i]  # 归一化第 i 个特征
            self.normalize = 0  # 内部归一化标志设置为 0
        else:
            R = self.R0
            Mreg = np.ones((d, 1)) * d
            self.normalize = 2  # 内部归一化标志设置为 2

        # 划分数据集，划分为训练集和验证集
        # 获得训练集和验证集的数据索引
        train = np.random.choice(n, int(n * self.split), replace=False)
        val = [i for i in np.arange(n) if i not in train]

        Train_R = R[train, :]  # 训练集的特征矩阵
        Val_R = R[val, :]  # 验证集的特征矩阵
        Train_y = self.Ut[train, :]  # 训练集的目标矩阵
        Val_y = self.Ut[val, :]  # 验证集的目标矩阵

        # 初始化容差 tol 和 l0 惩罚项
        d_tol = float(self.d_tol)
        # 判断是否为 STRidge 第一次迭代，若为首次迭代则初始化(self.it将在train函数中进行定义)
        if self.it == 0:
            self.tol = d_tol  # self.tol 即为当前容差(首次定义!!)

        # 利用 lambda1 参数和归一化矩阵 Mreg 进行调整得到初始权重 w_best
        w_best = self.lambda1 / Mreg

        # 计算验证集中预测值和真实值的真实误差(MSE)，R 和 w 做的是最小二乘法
        err_f = np.mean((Val_y - np.dot(Val_R, w_best))**2)

        # 初始化 l0 惩罚项相关系数
        if self.l0_penalty is None and self.it == 0:
            self.l0_penalty_0 = err_f  # 定义模型的 l0 惩罚项
            l0_penalty = self.l0_penalty_0  # 将模型的验证集误差作为惩罚系数
        elif self.l0_penalty is None:
            print("Warning! l0_penalty is None! Please input the parameter `l0_penalty_0` from the previous training")
            l0_penalty = self.l0_penalty_0

        # 计算 l0 惩罚项
        err_lambda = l0_penalty * np.count_nonzero(w_best)  # 计算 w_best 的非零值，与岭回归相关
        # 计算总误差
        err_best = err_f + err_lambda  # 定义最优误差，该行为初始化
        # 初始化最优容差为 0，即接受所有的项
        tol_best = 0

        # 进行优化
        for iter in range(self.maxit):
            # 获取当前最小二乘参数 w
            w = self.STRidge(Train_R, Train_y, self.lam, self.STR_iters, self.tol,
                             Mreg, normalize=self.normalize)

            # 计算验证集的物理信息误差
            err_f = np.mean((Val_y - np.dot(Val_R, w))**2)

            # 计算 l0 惩罚项
            err_lambda = l0_penalty * np.count_nonzero(w)

            # 计算总误差
            err = err_f + err_lambda

            # 检查并更新容差和误差
            if err <= err_best:
                # 总误差诶过减少就覆盖并记录
                err_best = err
                w_best = w  # 更新权重参数 w
                tol_best = self.tol  # 读取当前最优容差，在后面保存
                self.tol = self.tol + d_tol  # 增加容差，进一步训练缩小范围

            else:
                # 否则减少容差并调整更新容差
                self.tol = max([0, self.tol - 2 * d_tol])
                d_tol = d_tol / 1.618
                self.tol = self.tol + d_tol

        # 打印最优容差
        if self.print_best_tol:
            print("Err best: %e, err_f: %e, err_lambda: %e," % (err_best, err_f, err_lambda))
            print("Optimal tolerance:", tol_best)
            print("\n")

        # 返回最优系数
        return np.real(np.multiply(Mreg, w_best))


# 使模块级别的名称等同于类
sys.modules[__name__] = STRidge