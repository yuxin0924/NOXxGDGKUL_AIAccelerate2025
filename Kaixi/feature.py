import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

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
    df_actual = df_actual.rename(columns={'price_eur_mwh': 'actual_price'})
    print(f"  - 实际数据 (actual) 加载完毕，形态: {df_actual.shape}")

    # 3. 15分钟间隔的日前市场价格（一个重要特征）
    df_dam = pd.read_csv(
        'dam_prices.csv',
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


