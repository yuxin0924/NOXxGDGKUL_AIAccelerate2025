import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from datetime import datetime, timedelta
import requests
import sys
import os

# 忽略一些常见的 pandas 警告
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def load_realtime_data():
    """从API获取实时数据和最新的DAM价格"""
    print("获取实时数据...")
    
    # 1. 获取实时预测数据
    print("1. 获取实时预测数据...")
    forecast_url = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods161/records"
    r_forecast = requests.get(forecast_url, params={"limit": 20, "order_by": "datetime DESC"}, timeout=10)
    df_forecast = pd.DataFrame(r_forecast.json()["results"])[["datetime", "imbalanceprice"]]
    df_forecast.columns = ["datetime_utc", "price_eur_mwh"]
    df_forecast["datetime_utc"] = pd.to_datetime(df_forecast["datetime_utc"], utc=True)
    df_forecast = df_forecast.set_index('datetime_utc').sort_index()
    
    # 2. 获取历史实际数据
    print("2. 获取历史实际数据...")
    actual_url = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"
    r_actual = requests.get(actual_url, params={"limit": 20, "order_by": "datetime DESC"}, timeout=10)
    df_actual = pd.DataFrame(r_actual.json()["results"])[["datetime", "imbalanceprice"]]
    df_actual.columns = ["datetime_utc", "price_eur_mwh"]
    df_actual["datetime_utc"] = pd.to_datetime(df_actual["datetime_utc"], utc=True)
    df_actual = df_actual.set_index('datetime_utc').sort_index()
    
    # 3. 读取最新的DAM价格
    print("3. 读取DAM价格数据...")
    df_dam = pd.read_csv('data/dam_1028.csv')
    df_dam['datetime_utc'] = pd.to_datetime(df_dam['datetime_utc'])
    df_dam = df_dam.set_index('datetime_utc').sort_index()
    
    # 重命名列以保持一致性
    df_forecast = df_forecast.rename(columns={'price_eur_mwh': 'forecast_price'})
    df_actual = df_actual.rename(columns={'price_eur_mwh': 'target_actual_price'})
    df_dam = df_dam.rename(columns={'price_eur_mwh': 'dam_price'})
    return df_forecast, df_actual, df_dam

r = requests.get(
    "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods161/records",
    params={"limit": 10, "order_by": "datetime DESC"},
    timeout=10
)
df_forecast = pd.DataFrame(r.json()["results"])[["datetime", "imbalanceprice"]]
df_forecast.columns = ["datetime_utc", "price_eur_mwh"]

df_forecast["datetime_utc"] = pd.to_datetime(df_forecast["datetime_utc"], utc=True)
df_forecast["date"]   = df_forecast["datetime_utc"].dt.date
df_forecast["hour"]   = df_forecast["datetime_utc"].dt.hour
df_forecast["minute"] = df_forecast["datetime_utc"].dt.minute
df_forecast["second"] = df_forecast["datetime_utc"].dt.second

df_forecast = df_forecast[["datetime_utc","date","hour","minute","second","price_eur_mwh"]]

def load_training_data():
    """加载训练用的历史数据（CSV文件）。"""
    print("开始加载数据...")

    # 1. 1分钟间隔的预测不平衡价格（我们的主要特征来源）
    df_forecast = pd.read_csv("data/imbalance_forecast.csv")
    if 'datetime_utc' not in df_forecast.columns:
        raise KeyError("缺少 datetime_utc 列")
    df_forecast['datetime_utc'] = pd.to_datetime(df_forecast['datetime_utc'], errors='coerce', utc=True)
    df_forecast = df_forecast.dropna(subset=['datetime_utc'])
    df_forecast = df_forecast.set_index('datetime_utc').sort_index()
    # 重命名价格列以便区分
    df_forecast = df_forecast.rename(columns={'price_eur_mwh': 'forecast_price'})
    print(f"  - 预测数据 (forecast) 加载完毕，形态: {df_forecast.shape}")

    # 2. 15分钟间隔的实际不平衡价格（我们的目标变量来源）
    df_actual = pd.read_csv("data/imbalance_actual.csv")
    if 'datetime_utc' not in df_actual.columns:
        raise KeyError("缺少 datetime_utc 列")
    df_actual['datetime_utc'] = pd.to_datetime(df_actual['datetime_utc'], errors='coerce', utc=True)
    df_actual = df_actual.dropna(subset=['datetime_utc'])
    df_actual = df_actual.set_index('datetime_utc').sort_index()
    # 重命名价格列以便区分
    df_actual = df_actual.rename(columns={'price_eur_mwh': 'target_actual_price'})
    print(f"  - 实际数据 (actual) 加载完毕，形态: {df_actual.shape}")

    # 3. 15分钟间隔的日前市场价格（一个重要特征）
    df_dam = pd.read_csv('data/dam_prices.csv')
    if 'datetime_utc' not in df_dam.columns:
        raise KeyError("缺少 datetime_utc 列")
    df_dam['datetime_utc'] = pd.to_datetime(df_dam['datetime_utc'], errors='coerce', utc=True)
    df_dam = df_dam.dropna(subset=['datetime_utc'])
    df_dam = df_dam.set_index('datetime_utc').sort_index()
    df_dam = df_dam.rename(columns={'price_eur_mwh': 'dam_price'})
    print(f"  - 日前价格 (DAM) 加载完毕，形态: {df_dam.shape}")

    return df_forecast, df_actual, df_dam


def create_features_and_target(df_forecast, df_actual, df_dam, is_training=True):
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
    if is_training:
        # 训练模式：删除所有缺失值
        df = df.dropna()
    else:
        # 预测模式：只保留最后一行，允许目标变量为空
        df = df.tail(1).copy()
        # 只删除特征中的缺失值，允许目标变量为空
        feature_cols = [col for col in df.columns if col != 'target_actual_price']
        df = df.dropna(subset=feature_cols)

    print(f"特征工程完成。最终数据集形态: {df.shape}")

    # 快速展示处理后数据以便检视
    print("\n== 处理后数据快照 ==")
    print(f"形状: {df.shape}")
    print("\n前 10 行:")
    print(df.head(10))
    print("\n后 5 行:")
    print(df.tail(5))

    print("\n列与类型:")
    print(df.dtypes)
    print("\n每列缺失值数量:")
    print(df.isna().sum())

    print("\n数值列描述性统计:")
    pd.set_option('display.max_columns', None)  # ensure pandas prints all columns
    pd.set_option('display.max_rows', None)     # show all rows

    print("\nAll columns:")
    # Return the processed dataframe directly without saving to CSV
    return df

def train_model(df):
    """训练 LightGBM 模型并返回模型对象。"""
    
    # 定义目标和特征
    target_col = 'target_actual_price'
    cols_to_drop = ['date', 'second', 'target_timestamp', target_col, 'hour', 'minute']
    features = [col for col in df.columns if col not in cols_to_drop]
    categorical_features = ['dayofweek', 'month', 'quarter', 'is_weekend']
    
    X = df[features]
    y = df[target_col]
    
    # 按时间顺序分割数据
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.85)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:val_size], y.iloc[train_size:val_size]
    X_test, y_test = X.iloc[val_size:], y.iloc[val_size:]
    
    print(f"数据拆分完毕:")
    print(f"  - 训练集: {X_train.shape}")
    print(f"  - 验证集: {X_val.shape}")
    print(f"  - 测试集: {X_test.shape}")
    
    # 初始化模型
    model = lgb.LGBMRegressor(
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=True)],
        categorical_feature=categorical_features
    )
    
    # 评估模型
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n测试集 MAE: {mae:.4f}")
    
    # Return model and features directly without saving
    print("\n模型训练完成")
    return model, features

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


def send_prediction_email(current_price, predicted_price, recipient_email="kaixi.yao@outlook.com"):
    """发送预测结果邮件"""
    # 邮件配置
    smtp_server = "smtp.mail.yahoo.com"
    smtp_port = 587
    sender_email = "yao_philip@yahoo.com"  # 替换为您的 Outlook 邮箱
    app_password = "ulirkiciqhnrilet"  # 替换为您的应用专用密码

    try:
        # 计算价格变化
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100 if current_price != 0 else 0
        
        # 获取时间
        current_time = datetime.now()
        next_quarter = current_time + timedelta(minutes=(15 - current_time.minute % 15))
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['Subject'] = f'NOX Energy - 价格预测更新 ({next_quarter.strftime("%H:%M")} UTC)'
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        # 邮件正文
        body = f"""NOX Energy 不平衡电价预测

预测生成时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
预测目标时段: {next_quarter.strftime('%H:%M')} - {(next_quarter + timedelta(minutes=15)).strftime('%H:%M')} UTC
预测价格: {predicted_price:.2f} EUR/MWh
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Create prediction DataFrame and convert directly to CSV string
        df_prediction = pd.DataFrame({
            'timestamp': [current_time, next_quarter],
            'price_type': ['当前价格', '预测价格'],
            'price_eur_mwh': [current_price, predicted_price]
        })
        
        # Convert DataFrame to CSV string in memory
        csv_string = df_prediction.to_csv(index=False, encoding='utf-8')
        
        # Attach CSV string directly to email
        part = MIMEApplication(csv_string.encode('utf-8'), _subtype="csv")
        part.add_header('Content-Disposition', 'attachment', filename="price_prediction.csv")
        msg.attach(part)

        # 连接到SMTP服务器并发送邮件
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用TLS加密
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        print(f"✓ 预测报告已发送至 {recipient_email}")
        
        # Log prediction to console instead of file
        print("预测日志:")
        print(body)
        print("="*50)
        
    except Exception as e:
        print(f"发送邮件失败: {str(e)}")
        if isinstance(e, smtplib.SMTPAuthenticationError):
            print("认证失败。请检查邮箱和应用密码。")
        elif isinstance(e, smtplib.SMTPServerDisconnected):
            print("服务器连接断开。请检查网络连接。")

def predict_with_model(df_forecast, df_actual, df_dam, model, features):
    """使用提供的模型对实时数据进行预测"""
    if model is None or features is None:
        raise ValueError("未提供模型或特征列表")
    
    # 处理实时数据
    df = create_features_and_target(df_forecast, df_actual, df_dam, is_training=False)
    
    if df.empty:
        raise ValueError("处理后的特征数据为空")
    
    # 确保所有需要的特征都存在
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"缺少特征: {feature}")
    
    # 获取最新时间点的特征
    X_pred = df[features].tail(1)
    
    # 进行预测
    y_pred = model.predict(X_pred)
    
    prediction_time = X_pred.index[0]
    target_time = prediction_time + pd.Timedelta(minutes=(15 - prediction_time.minute % 15))
    
    result = {
        'prediction_time': prediction_time,
        'target_time': target_time,
        'current_price': df_forecast['forecast_price'].iloc[-1],
        'predicted_price': y_pred[0]
    }
    
    return result

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
    print("🔋 NOX Energy Imbalance Price Prediction 🔋")
    print("=========================================")
    current_time = datetime.now()
    print(f"当前时间 (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=========================================\n")
    
    try:
        # 加载训练数据并训练新模型
        print("\n1. 加载训练数据...")
        df_forecast, df_actual, df_dam = load_training_data()
        df_processed = create_features_and_target(df_forecast, df_actual, df_dam)
        
        if not df_processed.empty:
            print("\n2. 训练模型...")
            model, features = train_model(df_processed)
            print("✓ 模型训练完成")
        else:
            raise ValueError("处理后的训练数据为空")
        
        # 获取实时预测
        print("\n获取实时预测...")
        print("1. 加载实时数据...")
        rt_forecast, rt_actual, rt_dam = load_realtime_data()
        
        print("\n2. 使用模型预测...")
        result = predict_with_model(rt_forecast, rt_actual, rt_dam, model, features)
        
        print("\n=== 预测结果 ===")
        print(f"预测时间: {result['prediction_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"目标时间: {result['target_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"预测价格: {result['predicted_price']:.2f} EUR/MWh")
        
        # 发送邮件通知
        print("\n3. 发送预测结果邮件...")
        send_prediction_email(result['current_price'], result['predicted_price'])
        
    except Exception as e:
        print(f"\n发生错误:")
        print(f"详情: {str(e)}")
    finally:
        print("\n处理结束。")
