import pandas as pd
df = pd.read_csv("versions/dry_bean_v2.csv")
df.insert(0, "bean_id", range(len(df)))
df["event_timestamp"] = pd.Timestamp("2026-10-01", tz="UTC")
df.to_parquet("feature_repo/data/beans.parquet", index=False)
