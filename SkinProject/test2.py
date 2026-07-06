import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("data_smallRange.xlsx")

X = df.iloc[:, 0:5]
y_n = df.iloc[:, 5]
y_g = df.iloc[:, 6]

# =========================
# 2. 划分训练 / 测试集
# =========================
X_train, X_test, y_n_train, y_n_test = train_test_split(
    X, y_n, test_size=0.2, random_state=42
)

_, _, y_g_train, y_g_test = train_test_split(
    X, y_g, test_size=0.2, random_state=42
)

# =========================
# 3. XGBoost 模型
# =========================
model_n = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model_g = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model_n.fit(X_train, y_n_train)
model_g.fit(X_train, y_g_train)

# =========================
# 4. 预测与评估
# =========================
y_n_pred = model_n.predict(X_test)
y_g_pred = model_g.predict(X_test)

print("n* R2:", r2_score(y_n_test, y_n_pred))
print("n* RMSE:", mean_squared_error(y_n_test, y_n_pred, squared=False))

print("g10* R2:", r2_score(y_g_test, y_g_pred))
print("g10* RMSE:", mean_squared_error(y_g_test, y_g_pred, squared=False))
