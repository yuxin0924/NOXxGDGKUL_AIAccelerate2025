import requests
import pandas as pd

def get_latest_imbalance_forecast(limit=5, timeout=10):
    """
    从 Elia Open Data (ODS161) 拉取最新 n 条 1min 不平衡价格预测。
    返回 DataFrame[datetime (UTC), quarterhour (UTC), imbalance_price]
    """
    url = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods161/records"
    params = {
        "limit": limit,
        "order_by": "datetime DESC"
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json().get("results", [])
    if not data:
        return pd.DataFrame(columns=["datetime","quarterhour","imbalanceprice"])
    df = pd.DataFrame(data)
    # 统一成你熟悉的列名
    df = df.rename(columns={
        "datetime": "datetime_utc",
        "quarterhour": "quarterhour_utc",
        "imbalanceprice": "price_eur_mwh"
    })
    # 转成时间类型并按时间倒序
    for c in ["datetime_utc", "quarterhour_utc"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df[["datetime_utc","quarterhour_utc","price_eur_mwh"]].sort_values("datetime_utc", ascending=False)

df_fore = get_latest_imbalance_forecast(limit=10)
print("✅ 最新 ODS161 预测（倒序）：")
print(df_fore.head(10))

