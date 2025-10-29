import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings

# 忽略一些常见的 pandas 警告
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- 1. 数据加载 ---
def load_data():
    """加载所有三个 CSV 文件并设置时间索引。"""
    print("开始加载数据...")
    df_forecast = pd.read_csv(
        'data/imbalance_forecast.csv', 
        parse_dates=['datetime_utc'],
        index_col='datetime_utc'
    ).rename(columns={'price_eur_mwh': 'forecast_price'})

    df_actual = pd.read_csv(
        'data/imbalance_actual.csv',
        parse_dates=['datetime_utc'],
        index_col='datetime_utc'
    ).rename(columns={'price_eur_mwh': 'target_actual_price'})

    df_dam = pd.read_csv(
        'data/dam_prices.csv',
        parse_dates=['datetime_utc'],
        index_col='datetime_utc'
    ).rename(columns={'price_eur_mwh': 'dam_price'})
    
    print(f"  - 预测数据 (forecast) 加载完毕，形态: {df_forecast.shape}")
    print(f"  - 实际数据 (actual) 加载完毕，形态: {df_actual.shape}")
    print(f"  - 日前价格 (DAM) 加载完毕，形态: {df_dam.shape}")
    
    return df_forecast, df_actual, df_dam

# --- 2. 特征工程与目标对齐 ---
def create_features_and_target(df_forecast, df_actual, df_dam):
    """
    合并数据框，创建特征和目标变量。
    """
    print("开始进行特征工程和目标对齐...")
    
    df = df_forecast.copy()

    # 2a. 创建目标变量 (Y) - 预测 *下一个* 15分钟的实际价格
    df['target_timestamp'] = df.index.floor('15min') + pd.Timedelta('15min')
    df = pd.merge(
        df,
        df_actual[['target_actual_price']],
        left_on='target_timestamp',
        right_index=True,
        how='left'
    )
    
    # 2b. 合并特征 (X) - 日前 (DAM) 价格 (防数据泄漏)
    df = pd.merge_asof(
        df.sort_index(),
        df_dam[['dam_price']].sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    # 2c. 创建基于时间的特征
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # 2d. 清理数据
    # 移除 'date', 'second', 'minute', 'target_timestamp', 'quarter' (如果存在)
    cols_to_drop = ['date', 'second', 'minute', 'target_timestamp']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = df.dropna()
    
    print(f"特征工程完成。最终数据集形态: {df.shape}")
    
    return df

# --- 3. 数据缩放和序列化 ---
def scale_and_create_sequences(df, sequence_length=60):
    """
    对数据进行缩放，并将其转换为适用于 LSTM 的序列。
    sequence_length: 模型回看的时间窗口（分钟数）。
    """
    print(f"开始缩放数据并创建 {sequence_length} 分钟的序列...")
    
    # 确保目标列是第一列，这有助于之后的反向缩放
    target_col = 'target_actual_price'
    cols = [target_col] + [col for col in df.columns if col != target_col]
    df = df[cols]
    
    # 初始化缩放器
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # 创建序列
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        # X 是过去 'sequence_length' 分钟的所有特征
        X.append(scaled_data[i-sequence_length:i, :])
        # y 是当前时间点的目标（在 scaled_data 中的第 0 列）
        y.append(scaled_data[i, 0]) 
        
    X, y = np.array(X), np.array(y)
    
    print(f"序列创建完毕。X shape: {X.shape}, y shape: {y.shape}")
    
    # X shape 将是 [样本数, 60, 特征数]
    # y shape 将是 [样本数]
    
    return X, y, scaler

# --- 4. 数据拆分 ---
def split_data(X, y):
    """按时间顺序拆分数据为训练、验证和测试集。"""
    
    # 重要的：按时间拆分数据，不能随机打乱！
    # 我们使用 70% 训练, 15% 验证, 15% 测试
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.85)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]

    print(f"数据拆分完毕:")
    print(f"  - 训练集: {X_train.shape}, {y_train.shape}")
    print(f"  - 验证集: {X_val.shape}, {y_val.shape}")
    print(f"  - 测试集: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- 5. 构建 LSTM 模型 ---
def build_model(input_shape):
    """构建 Keras LSTM 模型。"""
    
    model = Sequential()
    
    # 第一个 LSTM 层
    # input_shape 是 (sequence_length, n_features)
    # return_sequences=True 是因为我们要堆叠另一个 LSTM 层
    model.add(LSTM(
        units=50, 
        return_sequences=True, 
        input_shape=input_shape
    ))
    model.add(Dropout(0.2))

    # 第二个 LSTM 层
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # 输出层
    # Dense(1) 因为我们只预测一个值（价格）
    model.add(Dense(units=1))

    # 编译模型
    # 我们直接优化 'mae'，与比赛标准一致
    model.compile(optimizer='adam', loss='mae')
    
    model.summary()
    return model

# --- 6. 训练和评估 ---
def train_and_evaluate(df_processed):
    """完整的训练和评估流程。"""
    
    # 定义序列长度（例如，回看60分钟）
    SEQUENCE_LENGTH = 60
    
    # 3. 缩放和序列化
    X, y, scaler = scale_and_create_sequences(df_processed, SEQUENCE_LENGTH)
    
    # 4. 拆分数据
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # 5. 构建模型
    # input_shape 是 (timesteps, features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # 定义早期停止
    early_stopping = EarlyStopping(
        monitor='val_loss', # 监控验证集的 loss (即 MAE)
        patience=10,        # 10 个 epoch 没有改善就停止
        restore_best_weights=True # 恢复到最佳模型权重
    )

    print("\n开始训练 LSTM 模型...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    print("模型训练完毕。")
    
    # 7. 评估
    print("\n在测试集上进行预测和评估...")
    y_pred_scaled = model.predict(X_test)
    
    # y_test 是 1D 数组，y_pred_scaled 是 2D 数组 [n_samples, 1]
    # 我们需要将它们变回 [n_samples, n_features] 的形状才能进行反向缩放
    
    # 获取测试集中的特征数量
    n_features = X_test.shape[2]
    
    # 创建一个空的数组，形状为 (n_samples, n_features)
    # 我们只关心第0列（我们的目标）
    
    # 反向缩放 y_test (真实值)
    y_test_inverse = np.zeros((len(y_test), n_features))
    y_test_inverse[:, 0] = y_test
    y_test_inverse = scaler.inverse_transform(y_test_inverse)[:, 0] # 只取第一列

    # 反向缩放 y_pred (预测值)
    y_pred_inverse = np.zeros((len(y_pred_scaled), n_features))
    y_pred_inverse[:, 0] = y_pred_scaled.flatten()
    y_pred_inverse = scaler.inverse_transform(y_pred_inverse)[:, 0] # 只取第一列
    
    # 计算最终的 MAE
    final_mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    
    print(f"\n======================================================")
    print(f"模型在测试集上的最终 MAE (反向缩放后): {final_mae:.4f} EUR/MWh")
    print(f"======================================================")

    # 绘制结果
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inverse, label='Actual Price', alpha=0.8)
    plt.plot(y_pred_inverse, label='Predicted Price', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title(f"测试集上的实际值 vs 预测值 (MAE: {final_mae:.4f})")
    plt.xlabel(f"时间步 (Time Steps, {SEQUENCE_LENGTH}分钟后开始)")
    plt.ylabel("价格 (EUR/MWh)")
    plt.show()

# --- 主执行流程 ---
if __name__ == "__main__":
    # 1. 加载
    df_forecast, df_actual, df_dam = load_data()
    # 2. 预处理
    df_processed = create_features_and_target(df_forecast, df_actual, df_dam)
    
    if not df_processed.empty:
        # 3, 4, 5, 6. 训练和评估
        train_and_evaluate(df_processed)
    else:
        print("处理后数据为空，请检查数据源和时间范围。")
