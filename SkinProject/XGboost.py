import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("data_smallRange.xlsx")
X = df.iloc[:, 0:5]    # h, K, beta, lambda_f, phi_f
y_n = df.iloc[:, 5]    # n*
y_g = df.iloc[:, 6]    # g10*

# =========================
# 2. 训练 / 测试集划分
# =========================
X_train, X_test, y_n_train, y_n_test, y_g_train, y_g_test = train_test_split(
    X, y_n, y_g,
    test_size=0.2,
    random_state=42
)


# =========================
# 3. XGBoost 回归模型
# =========================
model_n = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.5,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,        # 去正则
    reg_lambda=0.0,
    random_state=42
)

model_g = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.5,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

model_n.fit(X_train, y_n_train)
model_g.fit(X_train, y_g_train)

# =========================
# 4. 预测与评估
# =========================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return r2, rmse

# Train
r2_n_tr, rmse_n_tr = evaluate(y_n_train, model_n.predict(X_train))
r2_g_tr, rmse_g_tr = evaluate(y_g_train, model_g.predict(X_train))

# Test
r2_n_te, rmse_n_te = evaluate(y_n_test, model_n.predict(X_test))
r2_g_te, rmse_g_te = evaluate(y_g_test, model_g.predict(X_test))

print("n*:  R2 Train =", r2_n_tr, "RMSE Train =", rmse_n_tr)
print("n*:  R2 Test  =", r2_n_te, "RMSE Test  =", rmse_n_te)

print("g10*: R2 Train =", r2_g_tr, "RMSE Train =", rmse_g_tr)
print("g10*: R2 Test  =", r2_g_te, "RMSE Test  =", rmse_g_te)
