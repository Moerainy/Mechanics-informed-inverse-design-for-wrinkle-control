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
    hidden_layer_sizes=(20, 20),#两层隐藏层
    activation='relu',
    solver='adam',
    max_iter=7000,
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


##GA

from scipy.optimize import differential_evolution

#基于baseline参数，BPNN跑出来的n，g10，这里可以修改成中年组的
n_base = 4.880758769134141
g_base = 1.057553643314153

#想要达到的参数，这里用的论文里的，最好是改成本代码中BPNN跑出来的结果
n_target = 4.73*0.8
g_target = 1.05*1.2

def objective(x):
    """
    x = [h, K, beta, lambda_f, phi_f]
    """
    x = np.array(x).reshape(1, -1)

    # 标准化
    x_s = scaler_X.transform(x)

    # BPNN 预测
    y_s = bpnn.predict(x_s)
    y = scaler_Y.inverse_transform(y_s)

    n_pred = y[0, 0]
    g_pred = y[0, 1]

    # 相对误差平方（无量纲）
    loss = ((n_pred - n_target) / n_target)**2 \
         + ((g_pred - g_target) / g_target)**2

    return loss
#范围
bounds = [
    (0.001, 0.015),        # h
    (0.1,   0.3),          # K
    (0.0,   np.pi / 2),    # beta
    (0.7,   1.0),          # lambda_f
    (0.55,  0.85)          # phi_f
]

result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=300,
    popsize=20,
    tol=1e-6,
    seed=42
)

optimal_x = result.x

# 用 BPNN 再算一遍结果
x_s = scaler_X.transform(optimal_x.reshape(1,-1))
y_s = bpnn.predict(x_s)
y = scaler_Y.inverse_transform(y_s)

print("\n=== BPNN-GA Optimization Result ===")
print("Optimal parameters:")
print("h =", optimal_x[0])
print("K =", optimal_x[1])
print("beta =", optimal_x[2])
print("lambda_f =", optimal_x[3])
print("phi_f =", optimal_x[4])

print("\nPredicted n* =", y[0,0])
print("Predicted g10* =", y[0,1])

print("\nTarget n* =", n_target)
print("Target g10* =", g_target)

