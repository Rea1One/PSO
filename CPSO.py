import numpy as np
import random
import matplotlib.pyplot as plt
import time
from Arguments import *
from QPSO import QPSO


# 混沌粒子群算法
class CPSO:
    # 初始化参数
    def __init__(self, pN, max_iter, chaotic_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.pN = pN  # 粒子数量
        self.p = 0.5
        self.M = 10  # 出救点个数
        self.N = 5  # 资源种类数
        self.max_iter = max_iter  # 迭代次数
        self.chaotic_iter = chaotic_iter
        self.x = np.zeros((self.pN, self.M, self.N))  # 所有粒子的位置
        self.V = np.zeros((self.pN, self.M, self.N))  # 所有粒子的速度
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
                        # print(delta)
            elif j < self.M - 1:
                for k in range(j + 1, self.M):
                    x[k][i] = 0

    # 初始化种群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.M):
                for k in range(self.N):
                    self.x[i][j][k] = random.randint(0, info[j][k])
                    self.V[i][j][k] = random.uniform(-x0, x0)
            self.modify(self.x[i])
            self.pbest[i] = self.x[i]
            tmp = self.function(self.x[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.x[i]

    # 更新粒子位置
    def iterator(self, chaotic):
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
            for i in range(self.pN):
                flag = 0
                tempv = self.V[i]
                tempx = self.x[i]
                while flag == 0 or np.min(self.x[i]) < 0:  # 更新粒子速度和位置
                    self.V[i] = self.w * tempv + self.c1 * random.uniform(0, 1) * (
                            self.pbest[i] - tempx) + self.c2 * random.uniform(0, 1) * (
                                        self.gbest - tempx)
                    self.x[i] = (tempx + self.V[i]).astype(np.int)
                    flag = 1
                self.modify(self.x[i])  # 修改向量以满足条件
            fitness.append(self.fit)
            if chaotic:
                self.chaotic()
            print("iter_num =", t)
            print("gbest:\n", self.gbest.astype(np.int))
            print(np.sum(self.gbest, axis=0))
            print("fitness =", int(self.fit))  # 输出最优值
            print("-----------------")
        return fitness

    # 混沌优化
    def chaotic(self):
        # 若小于优化概率，则本次不混沌优化
        if random.uniform(0, 1) < self.p:
            return
        s = (info - self.gbest) / info
        for k in range(self.chaotic_iter):
            s = 4 * s * (1 - s)
            tempx = (info - s * info).astype(np.int)
            f = self.function(tempx)
            if f < self.fit:
                self.fit = f
                self.gbest = tempx
                break


# 算法执行
cpso = CPSO(pN=20, max_iter=100, chaotic_iter=10)


basic_start = time.time()
cpso.init_Population()
fitness1 = cpso.iterator(False)
basic_end = time.time()
print('基本粒子群算法耗时：', basic_end - basic_start)


chao_start = time.time()
cpso.init_Population()
fitness2 = cpso.iterator(True)
chao_end = time.time()
print('混沌粒子群算法耗时：', chao_end - chao_start)


# 算法执行
qpso = QPSO(pN=20, max_iter=100)
quan_start = time.time()
qpso.init_Population()
fitness3 = qpso.iterator()
quan_end = time.time()
print('量子粒子群算法耗时：', quan_end - quan_start)

# 画图
plt.figure(1)
plt.xlabel("Iterators", size=14)
plt.ylabel("Fitness", size=14)
t = np.array([t for t in range(0, 100)])
fitness1 = np.array(fitness1)
plt.plot(t, fitness1, color='b', linewidth=3)
s = np.array([s for s in range(0, 100)])
fitness2 = np.array(fitness2)
plt.plot(s, fitness2, color='r', linewidth=3)
u = np.array([u for u in range(0, 100)])
fitness3 = np.array(fitness3)
plt.plot(s, fitness3, color='g', linewidth=3)
plt.legend(['Basic', 'Chaotic', 'Quantum'])
plt.savefig('./Result.jpg')
plt.show()
