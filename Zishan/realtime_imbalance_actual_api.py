import requests, pandas as pd

r = requests.get(
    "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods162/records",
    
    params={"limit": 10, "order_by": "datetime DESC"},
    timeout=10
)
df = pd.DataFrame(r.json()["results"])[["datetime", "imbalanceprice"]]
df.columns = ["datetime_utc", "price_eur_mwh"]

df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
df["date"]   = df["datetime_utc"].dt.date
df["hour"]   = df["datetime_utc"].dt.hour
df["minute"] = df["datetime_utc"].dt.minute
df["second"] = df["datetime_utc"].dt.second

df = df[["datetime_utc", "date", "hour", "minute", "second", "price_eur_mwh"]] \
       .sort_values("datetime_utc", ascending=False) \
       .reset_index(drop=True)

print(df)
