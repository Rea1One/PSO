import numpy as np
import random
from Arguments import *


# 量子粒子群算法
class QPSO:
    # 初始化参数
    def __init__(self, pN, max_iter):
        self.alpha = 0.6
        self.pN = pN  # 粒子数量
        self.M = 10  # 出救点个数
        self.N = 5  # 资源种类数
        self.max_iter = max_iter  # 迭代次数
        self.x = np.zeros((self.pN, self.M, self.N))  # 所有粒子的位置
        self.pbest = np.zeros((self.pN, self.M, self.N))  # 个体经历的最佳位置
        self.gbest = np.zeros((1, self.M, self.N))  # 全局最佳位置
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值

    # 计算应急点第i时刻第j类资源的数量
    def amount(self, i, j, x):
        add = 0
        for k in range(self.M):
            if i > t[k]:
                add += x[k][j]
        return add - i * V[j]

    # 计算适应度
    def function(self, x):
        Z = 0
        for i in range(self.M):  # 计算运送资源的总成本
            for j in range(self.N):
                Z += C[i][j] * x[i][j]
        for i in range(max(t)):  # 计算资源缺失的总成本
            for j in range(self.N):
                temp = self.amount(i, j, x)
                if temp < 0:  # 若资源在第i时刻已缺失
                    Z += D[j] * (V[j] * (i + 0.5) - temp)
                elif self.amount(i + 1, j, x) < 0:  # 若资源在[i,i+1]时间段内缺失
                    tn = self.amount(i, j, x) / V[j] + 1
                    Z += D[j] * V[j] * ((i + 1) ** 2 - tn ** 2) / 2
        return Z

    # 修改向量以满足约束条件
    def modify(self, x):
        for i in range(self.M):
            for j in range(self.N):
                if 0 < x[i][j] < 2 or x[i][j] > info[i][j]:
                    x[i][j] = 0
        for i in range(self.N):
            temp = 0
            j = 0
            flag = 0  # 标识各列之和是否大于所需该类物资总数
            for j in range(self.M):
                if temp + x[j][i] >= X[i]:
                    x[j][i] = X[i] - temp
                    flag = 1
                    break
                temp += x[j][i]
            if j == self.M - 1 and flag == 0:
                delta = X[i] - temp
                if x[j][i] + delta <= info[j][i]:
                    x[j][i] += delta
                else:
                    k = 0
                    while delta > 0:
                        if x[k][i] == 0:
                            x[k][i] += x0
                            delta -= x0
                        elif x[k][i] + 1 <= info[k][i]:
                            x[k][i] += 1
                            delta -= 1
                        k = (k + 1) % self.M
            elif j < self.M - 1:
                for k in range(j + 1, self.M):
                    x[k][i] = 0

    # 初始化种群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.M):
                for k in range(self.N):
                    self.x[i][j][k] = random.randint(0, info[j][k])
            self.modify(self.x[i])
            self.pbest[i] = self.x[i]
            tmp = self.function(self.x[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.x[i]

    # 求群体平均最优位置
    def getMean(self):
        mbest = np.empty([self.M, self.N], dtype=int)
        for i in range(self.pN):
            for j in range(self.M):
                for k in range(self.N):
                    mbest[j][k] += self.pbest[i][j][k]
        return mbest / self.pN

    # 更新粒子位置
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):
                temp = self.function(self.x[i])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.x[i]
                    if self.p_fit[i] < self.fit:  # 更新全局最优
                        self.gbest = self.x[i]
                        self.fit = self.p_fit[i]
            mbest = self.getMean()
            for i in range(self.pN):
                fi = random.uniform(0, 1)
                Pi = fi * self.pbest[i] + (1 - fi) * self.gbest
                flag = 0
                while flag == 0 or np.min(self.x[i]) < 0:
                    u = random.uniform(0, 1)
                    tempx = self.x[i].copy()
                    if random.random() > 0.5:
                        self.x[i] = Pi + self.alpha * np.log(1 / u) * np.fabs(mbest - tempx)
                    else:
                        self.x[i] = Pi - self.alpha * np.log(1 / u) * np.fabs(mbest - tempx)
                    flag = 1
                self.modify(self.x[i])  # 修改向量以满足条件
            fitness.append(self.fit)
            print("iter_num =", t)
            print("gbest:\n", self.gbest.astype(np.int))
            print(np.sum(self.gbest, axis=0).astype(int))
            print("fitness =", int(self.fit))  # 输出最优值
            print("-----------------")
        return fitness


