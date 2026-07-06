import pandas as pd
import statsmodels.api as sm

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("data_smallRange.xlsx")


# 自变量（5 个结构参数）
X = df.iloc[:, 0:5]   # A–E 列
X = sm.add_constant(X)  # 加截距项

# 因变量
y_n = df.iloc[:, 5]   # F 列：n*
y_g = df.iloc[:, 6]   # G 列：g10*

# =========================
# 2. 回归：n*
# =========================
model_n = sm.OLS(y_n, X).fit()

# =========================
# 3. 回归：g10*
# =========================
model_g = sm.OLS(y_g, X).fit()

# =========================
# 4. 输出结果
# =========================
print("=== 回归结果：n* ===")
print(model_n.summary())

print("\n=== 回归结果：g10* ===")
print(model_g.summary())
