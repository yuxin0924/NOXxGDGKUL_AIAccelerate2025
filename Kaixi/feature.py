import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# 忽略一些常见的 pandas 警告
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


def load_data():
    """加载所有三个 CSV 文件并设置时间索引。"""
    print("开始加载数据...")

    # 1. 1分钟间隔的预测不平衡价格（我们的主要特征来源）
    df_forecast = pd.read_csv(
        "data/imbalance_forecast.csv",
        parse_dates=['datetime_utc'],
        index_col='datetime_utc'
    )
    # 重命名价格列以便区分
    df_forecast = df_forecast.rename(columns={'price_eur_mwh': 'forecast_price'})
    print(f"  - 预测数据 (forecast) 加载完毕，形态: {df_forecast.shape}")

    # 2. 15分钟间隔的实际不平衡价格（我们的目标变量来源）
    df_actual = pd.read_csv(
        "data/imbalance_actual.csv",
        parse_dates=['datetime_utc'],
        index_col='datetime_utc'
    )
    # 重命名价格列以便区分
    df_actual = df_actual.rename(columns={'price_eur_mwh': 'target_actual_price'})
    print(f"  - 实际数据 (actual) 加载完毕，形态: {df_actual.shape}")

    # 3. 15分钟间隔的日前市场价格（一个重要特征）
    df_dam = pd.read_csv(
        'data/dam_prices.csv',
        parse_dates=['datetime_utc'],
        index_col='datetime_utc'
    )
    df_dam = df_dam.rename(columns={'price_eur_mwh': 'dam_price'})
    print(f"  - 日前价格 (DAM) 加载完毕，形态: {df_dam.shape}")

    return df_forecast, df_actual, df_dam


def create_features_and_target(df_forecast, df_actual, df_dam):
    """
    将数据框合并，创建特征和目标变量。
    目标：每分钟进行预测，预测下一个15分钟的实际价格。
    """
    print("开始进行特征工程和目标对齐...")

    # 1. 将1分钟的预测数据作为我们的基础数据框
    df = df_forecast.copy()

    # 2. 创建目标变量 (Y)
    # 目标是预测 *下一个* 15分钟的实际价格
    # 我们将每个1分钟的时间戳映射到它 *之后* 的那个15分钟的开始时间
    # 例如：14:01 -> 14:15, 14:14 -> 14:15, 14:15 -> 14:30
    df['target_timestamp'] = df.index.floor('15min') + pd.Timedelta('15min')

    # 将目标价格（实际价格）合并起来
    df = pd.merge(
        df,
        df_actual[['target_actual_price']],
        left_on='target_timestamp',
        right_index=True,
        how='left'
    )

    # 3. 合并特征 (X) - 日前 (DAM) 价格
    # 我们使用 merge_asof 来获取在当前1分钟时间戳 *之前* 的 *最近* 的15分钟DAM价格
    # 这能防止数据泄漏（即使用未来的信息）
    df = pd.merge_asof(
        df.sort_index(),
        df_dam[['dam_price']].sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'  # 获取之前最近的值
    )

    # 4. 创建基于时间的特征
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # 5. 创建滞后 (Lag) 和 滚动 (Rolling) 特征
    # 使用1分钟预测价格来创建特征
    df['forecast_price_lag_1min'] = df['forecast_price'].shift(1)
    df['forecast_price_lag_15min'] = df['forecast_price'].shift(15)

    # 5分钟滚动平均
    df['forecast_rolling_mean_5min'] = df['forecast_price'].rolling(window=5).mean().shift(1)
    # 15分钟滚动标准差（波动性）
    df['forecast_rolling_std_15min'] = df['forecast_price'].rolling(window=15).std().shift(1)

    # 6. 交互特征
    # 预测价格与日前价格的差异
    df['forecast_vs_dam'] = df['forecast_price'] - df['dam_price']

    # 7. 清理数据
    # 删除没有目标值的行（例如，数据集末尾的预测数据还没有对应的实际数据）
    # 同时删除因创建滞后/滚动特征而产生的早期 NaN 值
    df = df.dropna()

    print(f"特征工程完成。最终数据集形态: {df.shape}")

    return df


def train_model(df):
    """训练 LightGBM 模型并评估。"""

    # 定义目标和特征
    target_col = 'target_actual_price'

    # 从 'date' 和 'second' 列中移除，因为它们可能不是有用的特征
    # 'target_timestamp' 只是一个辅助列，也移除
    cols_to_drop = [
        'date', 'second', 'target_timestamp', target_col,
        'hour', 'minute'  # 已经用作分类特征了
    ]

    # 在 df.columns 中存在的列才会被移除
    features = [col for col in df.columns if col not in cols_to_drop]

    # 定义分类特征，LGBM 可以更好地处理它们
    categorical_features = ['dayofweek', 'month', 'quarter', 'is_weekend']

    # 确保所有特征列都是LGBM支持的类型（例如，datetime索引需要移除）
    # 在这里，我们的特征都应该是数值型或已定义的分类

    X = df[features]
    y = df[target_col]

    # 重要的：按时间拆分数据，不能随机打乱！
    # 我们使用 70% 训练, 15% 验证, 15% 测试

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.85)

    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:val_size], y.iloc[train_size:val_size]
    X_test, y_test = X.iloc[val_size:], y.iloc[val_size:]

    print(f"数据拆分完毕:")
    print(f"  - 训练集: {X_train.shape}")
    print(f"  - 验证集: {X_val.shape}")
    print(f"  - 测试集: {X_test.shape}")

    # 初始化 LightGBM 模型
    # 目标是最小化 MAE (Mean Absolute Error)
    lgb_model = lgb.LGBMRegressor(
        objective='mae',
        metric='mae',
        n_estimators=1000,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )

    print("\n开始训练模型...")

    # 训练模型，并使用验证集进行早期停止
    # 这可以防止模型过拟合，并在验证集MAE不再改善时停止训练
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=True)],
        categorical_feature=categorical_features
    )

    print("模型训练完毕。")

    return lgb_model, X_test, y_test, features


def evaluate_and_predict(model, X_test, y_test, feature_names):
    """在测试集上进行预测并评估模型。"""

    print("\n在测试集上进行预测...")
    y_pred = model.predict(X_test)

    # 计算 MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n======================================")
    print(f"模型在测试集上的 MAE: {mae:.4f}")
    print(f"======================================")

    # 显示特征重要性
    print("\n显示特征重要性...")
    lgb.plot_importance(model, max_num_features=20, figsize=(10, 8), importance_type='gain')
    plt.title("LightGBM importance of feature (based on gain)")
    plt.tight_layout()
    plt.show()

    # (可选) 绘制预测值 vs 实际值
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='Predicted Price', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title(f"Real values from test set vs prediction values (MAE: {mae:.4f})")
    plt.xlabel("time")
    plt.ylabel("price (EUR/MWh)")
    plt.show()


if __name__ == "__main__":
    # 运行整个流程
    df_forecast, df_actual, df_dam = load_data()
    df_processed = create_features_and_target(df_forecast, df_actual, df_dam)

    if not df_processed.empty:
        model, X_test, y_test, features = train_model(df_processed)
        evaluate_and_predict(model, X_test, y_test, features)
    else:
        print("处理后数据为空，请检查数据源和时间范围。")
