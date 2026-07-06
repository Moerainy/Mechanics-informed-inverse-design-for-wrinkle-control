import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_excel("data_smallRange.xlsx")

X = df.iloc[:, 0:5]   # h, K, beta, lambda_f, phi_f
Y = df.iloc[:, 5:7]   # n*, g10*

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_s = scaler_X.fit_transform(X)
Y_s = scaler_Y.fit_transform(Y)

bpnn = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation='relu',
    solver='adam',
    alpha=1e-4,              # L2 正则，防过拟合（非常重要）
    learning_rate_init=1e-3,
    max_iter=15000,
    early_stopping=True,     # 自动用 validation
    validation_fraction=0.2,
    n_iter_no_change=50,
    random_state=42
)


bpnn.fit(X_s, Y_s)


Y_pred_s = bpnn.predict(X_s)
Y_pred = scaler_Y.inverse_transform(Y_pred_s)

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

# n*
r2_n, rmse_n = evaluate(Y.iloc[:,0], Y_pred[:,0])

# g10*
r2_g, rmse_g = evaluate(Y.iloc[:,1], Y_pred[:,1])

print("=== BPNN (100% Training Data) ===")
print(f"n*:   R2 = {r2_n:.4f}, RMSE = {rmse_n:.4f}")
print(f"g10*: R2 = {r2_g:.4f}, RMSE = {rmse_g:.4f}")

#中年组的base参数
new_X = np.array([[0.01, 0.225, np.pi/4, 0.93, 0.8]])

new_X_s = scaler_X.transform(new_X)
new_Y_s = bpnn.predict(new_X_s)
new_Y = scaler_Y.inverse_transform(new_Y_s)

print("\nPredicted n* for Middle-aged =", new_Y[0,0])
print("Predicted g10* for Middle-aged =", new_Y[0,1])

#老年组的base参数
new_X = np.array([[0.008, 0.175, np.pi/8, 0.96, 0.6]])

new_X_s = scaler_X.transform(new_X)
new_Y_s = bpnn.predict(new_X_s)
new_Y = scaler_Y.inverse_transform(new_Y_s)

print("\nPredicted n* for Elderly =", new_Y[0,0])
print("Predicted g10* for Elderly=", new_Y[0,1])


import numpy as np

# ============================================================
# 1. 目标参数（反演目标）
# ============================================================

n_base = 4.880758769134141
g_base = 1.057553643314153

n_target = 4.75290160098888*0.9
g_target = 1.0529135945021793*1.1


# ============================================================
# 2. 适应度函数
# ============================================================

def objective(x):
    x = np.array(x).reshape(1, -1)

    x_s = scaler_X.transform(x)
    y_s = bpnn.predict(x_s)
    y = scaler_Y.inverse_transform(y_s)

    n_pred, g_pred = y[0, 0], y[0, 1]

    loss = ((n_pred - n_target) / n_target) ** 2 \
         + ((g_pred - g_target) / g_target) ** 2

    return loss


# ============================================================
# 3. 编码方式 & 初始种群（来自数据集）
# ============================================================

# 五个设计参数
param_columns = df.columns[0:5]

# 初始种群：直接使用已有数据（10,200 个个体）
population = df[param_columns].values.copy()
pop_size = population.shape[0]          # 10200
n_gene = population.shape[1]            # 5


# ============================================================
# 4. GA 参数
# ============================================================

mutation_rate = 0.1
mutation_size = int(0.1 * pop_size)     # 1020
max_stagnation = 100                     # 连续 100 代无改进


# ============================================================
# 5. GA 算子
# ============================================================

def crossover(pop):
    """两两交叉，单点交叉"""
    offspring = []

    idx = np.random.permutation(len(pop))
    for i in range(0, len(pop), 2):
        p1 = pop[idx[i]]
        p2 = pop[idx[i+1]]

        cp = np.random.randint(1, n_gene)   # 随机交叉点
        c1 = np.concatenate([p1[:cp], p2[cp:]])
        c2 = np.concatenate([p2[:cp], p1[cp:]])

        offspring.append(c1)
        offspring.append(c2)

    return np.array(offspring[:pop_size])


def mutation(pop, bounds):
    mutants = []
    for _ in range(mutation_size):
        ind = pop[np.random.randint(len(pop))].copy()
        gene = np.random.randint(n_gene)

        low, high = bounds[gene]
        ind[gene] = np.random.uniform(low, high)

        mutants.append(ind)

    return np.array(mutants)


def select_elite(pool, pool_fitness):
    idx = np.argsort(pool_fitness)
    return pool[idx[:pop_size]]


# ============================================================
# 6. 参数范围（用于变异）
# ============================================================

bounds = np.array([
    [0.001, 0.015],        # h
    [0.1,   0.3],          # K
    [0.0,   np.pi/2],      # beta
    [0.7,   1.0],          # lambda_f
    [0.55,  0.85]          # phi_f
])


# ============================================================
# 7. 主进化循环（精英保留）
# ============================================================

best_loss = np.inf
best_individual = None
stagnation = 0
generation = 0

while stagnation < max_stagnation:

    # 当前种群适应度
    fitness_pop = np.array([objective(ind) for ind in population])

    # 当前最优
    gen_best = fitness_pop.min()
    if gen_best < best_loss:
        best_loss = gen_best
        best_individual = population[np.argmin(fitness_pop)]
        stagnation = 0
    else:
        stagnation += 1

    # 交叉
    offspring = crossover(population)

    # 变异
    mutants = mutation(population, bounds)

    # 合并池：父代 + 子代 + 变异体
    pool = np.vstack([population, offspring, mutants])
    pool_fitness = np.array([objective(ind) for ind in pool])

    # 精英选择
    population = select_elite(pool, pool_fitness)

    generation += 1
    if generation % 10 == 0:
        print(f"Gen {generation}, best loss = {best_loss:.4e}")

# ============================================================
# 8. 最终结果
# ============================================================

optimal_x = best_individual

x_s = scaler_X.transform(optimal_x.reshape(1, -1))
y_s = bpnn.predict(x_s)
y = scaler_Y.inverse_transform(y_s)

print("\n=== BPNN + GA Inverse Design Result ===")
print("Optimal parameters:")
print("h =", optimal_x[0])
print("K =", optimal_x[1])
print("beta =", optimal_x[2])
print("lambda_f =", optimal_x[3])
print("phi_f =", optimal_x[4])

print("\nPredicted n* =", y[0, 0])
print("Predicted g10* =", y[0, 1])

print("\nTarget n* =", n_target)
print("Target g10* =", g_target)

