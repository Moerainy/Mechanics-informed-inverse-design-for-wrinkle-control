import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# =====================================================
# 1. 读取数据
# =====================================================
df = pd.read_excel("data_smallRange.xlsx")

X = df.iloc[:, 0:5].values   # h, K, beta, lambda_f, phi_f
Y = df.iloc[:, 5:7].values   # n*, g10*

# =====================================================
# 2. 划分：训练 / 验证 / 测试 (70 / 15 / 15)
# =====================================================
# 70% 训练集，30% 非训练集
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.30, random_state=42
)

# 30% 非训练集再平分成 15% 验证集 + 15% 测试集
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.50, random_state=42
)

# =====================================================
# 3. 标准化（只用训练集 fit）
# =====================================================
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
Y_train_s = scaler_Y.fit_transform(Y_train)

X_val_s = scaler_X.transform(X_val)
Y_val_s = scaler_Y.transform(Y_val)

X_test_s = scaler_X.transform(X_test)
Y_test_s = scaler_Y.transform(Y_test)

X_all_s = scaler_X.transform(X)
Y_all_s = scaler_Y.transform(Y)

# =====================================================
# 4. BPNN 模型
# =====================================================
bpnn = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    solver='adam',
    max_iter=7000,
    random_state=42
)

bpnn.fit(X_train_s, Y_train_s)

# =====================================================
# 5. 预测
# =====================================================
def predict_and_inverse(X_s):
    Y_s = bpnn.predict(X_s)
    return scaler_Y.inverse_transform(Y_s)

Y_train_pred = predict_and_inverse(X_train_s)
Y_val_pred   = predict_and_inverse(X_val_s)
Y_test_pred  = predict_and_inverse(X_test_s)
Y_all_pred   = predict_and_inverse(X_all_s)

# =====================================================
# 6. 评估函数
# =====================================================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

def print_metrics(name, Y_true, Y_pred):
    r2_n, rmse_n = evaluate(Y_true[:,0], Y_pred[:,0])
    r2_g, rmse_g = evaluate(Y_true[:,1], Y_pred[:,1])
    print(f"\n=== {name} ===")
    print(f"n*:   R2 = {r2_n:.4f}, RMSE = {rmse_n:.4f}")
    print(f"g10*: R2 = {r2_g:.4f}, RMSE = {rmse_g:.4f}")

# =====================================================
# 7. 输出结果
# =====================================================
print_metrics("Training Set", Y_train, Y_train_pred)
print_metrics("Validation Set", Y_val, Y_val_pred)
print_metrics("Test Set", Y_test, Y_test_pred)
print_metrics("All Data", Y, Y_all_pred)

# =====================================================
# 8. Prediction vs Target 综合对比图（一张图）
# =====================================================
def plot_single_set(Y_true, Y_pred, title):

    plt.figure(figsize=(6, 6))

    # ===== scatter =====
    plt.scatter(
        Y_true[:, 0], Y_pred[:, 0],
        alpha=0.7, label='n*'
    )
    plt.scatter(
        Y_true[:, 1], Y_pred[:, 1],
        alpha=0.7, label='g10*'
    )

    # ===== y = x =====
    min_val = min(Y_true.min(), Y_pred.min())
    max_val = max(Y_true.max(), Y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle='--',
        linewidth=1
    )

    # ===== metrics =====
    r2_n = r2_score(Y_true[:, 0], Y_pred[:, 0])
    rmse_n = np.sqrt(mean_squared_error(Y_true[:, 0], Y_pred[:, 0]))
    r2_g = r2_score(Y_true[:, 1], Y_pred[:, 1])
    rmse_g = np.sqrt(mean_squared_error(Y_true[:, 1], Y_pred[:, 1]))

    # ===== title =====
    plt.title(title, fontsize=13)

    # ===== annotation =====
    plt.text(
        0.05, 0.95,
        f"n*:  R² = {r2_n:.3f}, RMSE = {rmse_n:.3f}\n"
        f"g10*: R² = {r2_g:.3f}, RMSE = {rmse_g:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.85
        )
    )

    plt.xlabel("Target Value")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# ===== 调用画图函数 =====
datasets = [
    (Y_train, Y_train_pred),
    (Y_val,   Y_val_pred),
    (Y_test,  Y_test_pred),
    (Y,       Y_all_pred)
]

titles = [
    "Training Set",
    "Validation Set",
    "Test Set",
    "All Data"
]

plot_single_set(Y_train, Y_train_pred, "Training Set")
plot_single_set(Y_val,   Y_val_pred,   "Validation Set")
plot_single_set(Y_test,  Y_test_pred,  "Test Set")
plot_single_set(Y,       Y_all_pred,   "All Data")



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

