import numpy as np
import random

NP = 20              # 种群规模
F = 0.8              # 缩放因子
CR = 0.9             # 交叉概率
max_gen = 1000       # 最大迭代代数
body_weight = 70.0   # 用户体重（kg）

dim = 8              # 决策向量维度

# 离散变量取值列表
cardio_freq_list   = [2, 3, 4, 5]    # 有氧训练频率（次/周）
cardio_dur_list    = [30, 45, 60]    # 有氧训练时长（分钟/次）
strength_freq_list = [2, 3, 4, 5]    # 力量训练频率（次/周）

# 决策向量解码
def decode(individual):
    C      = individual[0]
    P      = individual[1]
    H      = individual[2]
    FatPct = individual[3]
    cf_idx = int(round(individual[4]))
    cd_idx = int(round(individual[5]))
    sf_idx = int(round(individual[6]))
    sleep  = individual[7]
    return {
        'Calorie'       : C,
        'ProteinPct'    : P,
        'CarbsPct'      : H,
        'FatPct'        : FatPct,
        'CardioFreq'    : cardio_freq_list[cf_idx],
        'CardioDur'     : cardio_dur_list[cd_idx],
        'StrengthFreq'  : strength_freq_list[sf_idx],
        'SleepHours'    : sleep,
    }

#  适应度函数
def fitness(X):
    # 解码离散变量
    cardio_freq  = cardio_freq_list[int(round(X[4]))]
    cardio_dur   = cardio_dur_list[int(round(X[5]))]
    strength_freq= strength_freq_list[int(round(X[6]))]

    # 计算每日蛋白质摄入量 (g)
    protein_g = X[0] * (X[1] / 100.0) / 4.0 * 1000.0
    req_protein = 1.2 * body_weight * 1000.0
    muscle_loss_rate = max(0.0, (req_protein - protein_g) / req_protein)

    # 计算每周能量赤字 (kcal)
    # 先假设基础代谢率 BMR=body_weight*24*1 kcal/h
    BMR = body_weight * 24 * 1.0
    TDEE = BMR * 1.2 + cardio_freq * (cardio_dur * 5)  # 运动消耗约 5kcal/min
    weekly_deficit = max(0.0, (TDEE - X[0]) * 7)
    fat_loss_kg = weekly_deficit / 7700.0
    fat_loss_rate = fat_loss_kg

    # 可持续性评分（越大越难坚持）
    sustain_score = max(0, cardio_freq - 3) + max(0, strength_freq - 3) + max(0, 7 - X[7])

    # 权重分配
    w1, w2, w3 = 0.4, 0.4, 0.2
    return w1 * muscle_loss_rate - w2 * fat_loss_rate + w3 * sustain_score

# 初始化种群
def initialize_population():
    pop = []
    for _ in range(NP):
        C      = np.random.uniform(1500, 2200)
        P      = np.random.uniform(25, 45)
        H      = np.random.uniform(30, 50)
        F_pct  = 100.0 - P - H
        F_pct  = np.clip(F_pct, 20, 35)
        cf_idx = random.randint(0, len(cardio_freq_list)-1)
        cd_idx = random.randint(0, len(cardio_dur_list)-1)
        sf_idx = random.randint(0, len(strength_freq_list)-1)
        sleep  = np.random.uniform(6.5, 8.5)
        ind = np.array([C, P, H, F_pct, cf_idx, cd_idx, sf_idx, sleep], dtype=float)
        pop.append(ind)
    return np.array(pop)

# 差分进化核心算法
def differential_evolution():
    pop = initialize_population()
    fit = np.array([fitness(ind) for ind in pop])
    best_idx = np.argmin(fit)
    best = pop[best_idx].copy()

    for gen in range(1, max_gen+1):
        for i in range(NP):
            # 选取三个不同于 i 的个体
            idxs = list(range(NP))
            idxs.remove(i)
            r1, r2, r3 = random.sample(idxs, 3)

            # 变异
            V = pop[r1] + F * (pop[r2] - pop[r3])
            # 边界处理
            V[0] = np.clip(V[0], 1500, 2200)
            V[1] = np.clip(V[1], 25, 45)
            V[2] = np.clip(V[2], 30, 50)
            V[3] = np.clip(100 - V[1] - V[2], 20, 35)
            V[7] = np.clip(V[7], 6.5, 8.5)
            V[4] = np.clip(round(V[4]), 0, len(cardio_freq_list)-1)
            V[5] = np.clip(round(V[5]), 0, len(cardio_dur_list)-1)
            V[6] = np.clip(round(V[6]), 0, len(strength_freq_list)-1)

            # 交叉
            U = pop[i].copy()
            j_rand = random.randint(0, dim-1)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    U[j] = V[j]

            # 选择
            if fitness(U) < fitness(pop[i]):
                pop[i] = U
                fit[i] = fitness(U)
                if fit[i] < fit[best_idx]:
                    best_idx = i
                    best = pop[i].copy()

        # 可选：打印进度
        if gen % 100 == 0:
            print(f"Generation {gen}, Best fitness = {fit[best_idx]:.4f}")

    return best, fit[best_idx]

if __name__ == "__main__":
    best_ind, best_val = differential_evolution()
    decoded = decode(best_ind)
    print("最优决策向量（原始表示）:", best_ind)
    print("最优解解码后方案:", decoded)
    print(f"最优适应度值: {best_val:.4f}")
