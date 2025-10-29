from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ================== 0️⃣ 路径设定（关键修复） ==================
# 以脚本所在目录为基准，向上一级就是项目根目录（包含 data 和 Zishan）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"   # -> .../NOXxGDGKUL_AIAccelerate2025/data

dam_path = DATA_DIR / "dam_prices.csv"
imb_f_path = DATA_DIR / "imbalance_forecast.csv"
imb_a_path = DATA_DIR / "imbalance_actual.csv"

# 文件存在性检查，给出清晰错误信息
missing = [p for p in [dam_path, imb_f_path, imb_a_path] if not p.exists()]
if missing:
    print("❌ 找不到以下数据文件：")
    for p in missing:
        print("  -", p)
    print("\n请确认你的项目结构如下（data 与 Zishan 同级）：")
    print(f"{PROJECT_ROOT}")
    print("├── data")
    print("│   ├── dam_prices.csv")
    print("│   ├── imbalance_forecast.csv")
    print("│   └── imbalance_actual.csv")
    print("└── Zishan")
    print("    └── test.py")
    sys.exit(1)

# ================== 1️⃣ 读取数据 ==================
# 如果 CSV 里列名不同（例如 datetime_utc / price_eur_mwh），下面会统一处理
dam = pd.read_csv(dam_path, parse_dates=["datetime_utc"])
imb_f = pd.read_csv(imb_f_path, parse_dates=["datetime_utc"])
imb_a = pd.read_csv(imb_a_path, parse_dates=["datetime_utc"])

# 统一列名
dam = dam.rename(columns={"price_eur_mwh": "dam_price"})[["datetime_utc", "dam_price"]]
imb_f = imb_f.rename(columns={"price_eur_mwh": "forecast_price"})[["datetime_utc", "forecast_price"]]
imb_a = imb_a.rename(columns={"price_eur_mwh": "actual_price"})[["datetime_utc", "actual_price"]]

# ================== 2️⃣ 预处理：对齐时间 ==================
imb_f["datetime_utc"] = pd.to_datetime(imb_f["datetime_utc"], utc=True)
imb_a["datetime_utc"] = pd.to_datetime(imb_a["datetime_utc"], utc=True)
dam["datetime_utc"] = pd.to_datetime(dam["datetime_utc"], utc=True)

# forecast 从 1 分钟聚合成 15 分钟均值
imb_f["datetime_utc_15"] = imb_f["datetime_utc"].dt.floor("15min")
imb_f_15 = (imb_f.groupby("datetime_utc_15", as_index=False)["forecast_price"].mean()
                 .rename(columns={"datetime_utc_15": "datetime_utc"}))

# 合并成一个完整表（outer 确保时间轴完整）
df = (imb_a.merge(dam, on="datetime_utc", how="outer")
           .merge(imb_f_15, on="datetime_utc", how="outer"))
df = df.sort_values("datetime_utc").reset_index(drop=True)

# 填充外生特征（不能填充 target）
df["dam_price"] = df["dam_price"].ffill()
df["forecast_price"] = df["forecast_price"].ffill()

# ================== 3️⃣ 特征工程 ==================
df["hour"] = df["datetime_utc"].dt.hour
df["weekday"] = df["datetime_utc"].dt.weekday
df["month"] = df["datetime_utc"].dt.month
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# 添加滞后特征
for col in ["actual_price", "dam_price", "forecast_price"]:
    for lag in [1, 2, 3, 4]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

# 删除存在 NaN 的前几行（保证训练所需列齐全）
needed_cols = ["actual_price", "dam_price", "forecast_price"] + \
              [f"actual_price_lag{i}" for i in [1,2,3,4]] + \
              [f"dam_price_lag{i}" for i in [1,2,3,4]] + \
              [f"forecast_price_lag{i}" for i in [1,2,3,4]]
df = df.dropna(subset=needed_cols)

# ================== 4️⃣ 划分数据（按时间顺序） ==================
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

features = [
    "dam_price","forecast_price","hour","weekday","month","is_weekend",
    "actual_price_lag1","actual_price_lag2","actual_price_lag3","actual_price_lag4",
    "dam_price_lag1","dam_price_lag2","dam_price_lag3","dam_price_lag4",
    "forecast_price_lag1","forecast_price_lag2","forecast_price_lag3","forecast_price_lag4",
]

X_train, y_train = train[features], train["actual_price"]
X_test, y_test = test[features], test["actual_price"]

# ================== 5️⃣ 训练模型 ==================
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ================== 6️⃣ 评估 ==================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE  = {mae:.2f} EUR/MWh")
print(f"RMSE = {rmse:.2f} EUR/MWh")

# ================== 7️⃣ 可视化 ==================
plt.figure(figsize=(12, 5))
plt.plot(test["datetime_utc"].to_numpy(), y_test.to_numpy(), label="Actual", linewidth=2)
plt.plot(test["datetime_utc"].to_numpy(), y_pred, label="Predicted", alpha=0.7)
plt.title("Actual vs Predicted Imbalance Prices (Test set)")
plt.xlabel("Datetime (UTC)")
plt.ylabel("EUR/MWh")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 前 10 行对比
comparison = pd.DataFrame({
    "datetime_utc": test["datetime_utc"].values,
    "actual": y_test.values,
    "predicted": y_pred
})
print(comparison.head(10).to_string(index=False))
