import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 读取数据
df = pd.read_excel("data_smallRange.xlsx")

X = df.iloc[:, 0:5]   # h, K, beta, lambda_f, phi_f
y_n = df.iloc[:, 5]   # n*
y_g = df.iloc[:, 6]   # g10*


scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


X_train, X_test, y_n_train, y_n_test = train_test_split(
    X_scaled, y_n, test_size=0.2
)

_, _, y_g_train, y_g_test = train_test_split(
    X_scaled, y_g, test_size=0.2
)


bp_n = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=None
)

bp_g = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=None
)

bp_n.fit(X_train, y_n_train)
bp_g.fit(X_train, y_g_train)


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return r2, rmse

# n*
r2_n_tr, rmse_n_tr = evaluate(y_n_train, bp_n.predict(X_train))
r2_n_te, rmse_n_te = evaluate(y_n_test,  bp_n.predict(X_test))

# g10*
r2_g_tr, rmse_g_tr = evaluate(y_g_train, bp_g.predict(X_train))
r2_g_te, rmse_g_te = evaluate(y_g_test,  bp_g.predict(X_test))

print("=== BPNN Results ===")
print(f"n*:   R2 Train = {r2_n_tr:.4f}, RMSE Train = {rmse_n_tr:.4f}")
print(f"n*:   R2 Test  = {r2_n_te:.4f}, RMSE Test  = {rmse_n_te:.4f}")
print(f"g10*: R2 Train = {r2_g_tr:.4f}, RMSE Train = {rmse_g_tr:.4f}")
print(f"g10*: R2 Test  = {r2_g_te:.4f}, RMSE Test  = {rmse_g_te:.4f}")


#GA

bounds = []
for i in range(5):
    bounds.append((X.iloc[:, i].min(), X.iloc[:, i].max()))
bounds = np.array(bounds)


def fitness(x):
    x = np.array(x).reshape(1, -1)
    x_scaled = scaler_X.transform(x)
    n_pred = bp_n.predict(x_scaled)[0]
    return n_pred


import random

def genetic_algorithm(
    fitness_func,
    bounds,
    pop_size=40,
    generations=80,
    mutation_rate=0.1
):
    dim = bounds.shape[0]

    # 初始化种群
    population = np.random.uniform(
        bounds[:, 0], bounds[:, 1],
        size=(pop_size, dim)
    )

    for gen in range(generations):
        fitness_vals = np.array([fitness_func(ind) for ind in population])

        # 选择（保留前 50%）
        idx = np.argsort(fitness_vals)
        population = population[idx[:pop_size // 2]]

        # 交叉 + 变异
        offspring = []
        while len(offspring) < pop_size // 2:
            p1, p2 = population[np.random.randint(len(population), size=2)]
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2

            # 变异
            if np.random.rand() < mutation_rate:
                j = np.random.randint(dim)
                child[j] = np.random.uniform(bounds[j, 0], bounds[j, 1])

            offspring.append(child)

        population = np.vstack([population, offspring])

    best = min(population, key=fitness_func)
    return best, fitness_func(best)


best_x, best_n = genetic_algorithm(fitness, bounds)

print("=== GA Optimization Result (BPNN-GA) ===")
print("Optimal parameters:", best_x)
print("Predicted n*:", best_n)
